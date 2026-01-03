// src/autograd/metal/kernels.mm
// Metal Compute Shaders C++ Wrapper Implementation
// Copyright (c) 2025 GradFlow Project

#include "gradflow/autograd/metal/kernels.hpp"

#include "gradflow/autograd/metal/device.hpp"

#import <Metal/Metal.h>

#include <stdexcept>
#include <string>

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace gradflow {
namespace gpu {

// ===================================================================
// MetalKernelsImpl: Internal implementation class
// ===================================================================

class MetalKernelsImpl {
public:
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    id<MTLLibrary> library;

    // Compute Pipeline States
    id<MTLComputePipelineState> add_pipeline;
    id<MTLComputePipelineState> mul_pipeline;
    id<MTLComputePipelineState> sub_pipeline;
    id<MTLComputePipelineState> div_pipeline;
    id<MTLComputePipelineState> relu_pipeline;
    id<MTLComputePipelineState> sum_stage1_pipeline;
    id<MTLComputePipelineState> sum_stage2_pipeline;
    id<MTLComputePipelineState> mean_pipeline;

    MetalKernelsImpl(MetalDevice* metal_device) {
        device = (id<MTLDevice>)metal_device->getMetalDevice();
        command_queue = (id<MTLCommandQueue>)metal_device->getMetalCommandQueue();

        NSError* error = nil;

        // 優先順位 1: 事前コンパイルされた .metallib をロード（デプロイメント向け）
#ifdef GRADFLOW_METAL_LIB_PATH
        NSString* metallibPath = @GRADFLOW_METAL_LIB_PATH;
        library = [device newLibraryWithFile:metallibPath error:&error];

        if (!library) {
            error = nil;  // エラーをリセットして次のフォールバックへ
        }
#endif

        // 優先順位 2: ソースディレクトリの .metal ファイルをロード（開発環境向け）
        if (!library) {
            const char* source_path = __FILE__;  // kernels.mm path
            std::string source_dir(source_path);
            size_t last_slash = source_dir.find_last_of("/\\");
            if (last_slash != std::string::npos) {
                source_dir = source_dir.substr(0, last_slash);
            }
            std::string metal_source_path = source_dir + "/kernels.metal";

            NSString* sourcePath = [NSString stringWithUTF8String:metal_source_path.c_str()];
            NSString* sourceCode = [NSString stringWithContentsOfFile:sourcePath
                                                             encoding:NSUTF8StringEncoding
                                                                error:&error];

            if (sourceCode) {
                // Runtime コンパイル
                library = [device newLibraryWithSource:sourceCode options:nil error:&error];
            }
        }

        // 優先順位 3: デフォルトライブラリ（アプリケーションバンドルに含まれる場合）
        if (!library) {
            library = [device newDefaultLibrary];
        }

        // すべてのフォールバックが失敗
        if (!library) {
            std::string error_msg = "Failed to load Metal library. Tried:\n";
#ifdef GRADFLOW_METAL_LIB_PATH
            error_msg += "1. Precompiled .metallib: " GRADFLOW_METAL_LIB_PATH "\n";
#endif
            error_msg += "2. Runtime compile from source\n";
            error_msg += "3. Default library (app bundle)\n";
            if (error) {
                error_msg +=
                    "Last error: " + std::string([[error localizedDescription] UTF8String]);
            }
            throw std::runtime_error(error_msg);
        }

        // Create compute pipeline states
        add_pipeline = createPipeline("add_kernel");
        mul_pipeline = createPipeline("mul_kernel");
        sub_pipeline = createPipeline("sub_kernel");
        div_pipeline = createPipeline("div_kernel");
        relu_pipeline = createPipeline("relu_kernel");
        sum_stage1_pipeline = createPipeline("sum_kernel_stage1");
        sum_stage2_pipeline = createPipeline("sum_kernel_stage2");
        mean_pipeline = createPipeline("mean_kernel");
    }

    ~MetalKernelsImpl() {
        [mean_pipeline release];
        [sum_stage2_pipeline release];
        [sum_stage1_pipeline release];
        [relu_pipeline release];
        [div_pipeline release];
        [sub_pipeline release];
        [mul_pipeline release];
        [add_pipeline release];
        [library release];
    }

private:
    id<MTLComputePipelineState> createPipeline(const char* function_name) {
        NSError* error = nil;
        id<MTLFunction> function = [library newFunctionWithName:@(function_name)];
        if (!function) {
            throw std::runtime_error(std::string("Failed to find function: ") + function_name);
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                     error:&error];
        [function release];

        if (!pipeline) {
            throw std::runtime_error(std::string("Failed to create pipeline for ") + function_name +
                                     ": " + std::string([[error localizedDescription] UTF8String]));
        }

        return pipeline;
    }
};

// ===================================================================
// Helper Functions
// ===================================================================

/**
 * @brief Dispatch elementwise kernel
 */
static void dispatchElementwiseKernel(id<MTLComputePipelineState> pipeline,
                                      id<MTLCommandQueue> queue,
                                      id<MTLBuffer> buffer_a,
                                      id<MTLBuffer> buffer_b,
                                      id<MTLBuffer> buffer_c,
                                      uint32_t size) {
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_a offset:0 atIndex:0];
        [encoder setBuffer:buffer_b offset:0 atIndex:1];
        [encoder setBuffer:buffer_c offset:0 atIndex:2];
        [encoder setBytes:&size length:sizeof(uint32_t) atIndex:3];

        // Thread group size: 256
        NSUInteger threadGroupSize = 256;
        NSUInteger numThreadGroups = (size + threadGroupSize - 1) / threadGroupSize;

        MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize numGroups = MTLSizeMake(numThreadGroups, 1, 1);

        [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

// ===================================================================
// MetalKernels Public Implementation
// ===================================================================

MetalKernels::MetalKernels(MetalDevice* device)
    : impl_(std::make_unique<MetalKernelsImpl>(device)) {}

MetalKernels::~MetalKernels() = default;

void MetalKernels::add(const float* a, const float* b, float* c, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = impl_->device;

        // Wrap existing memory as MTLBuffer (Unified Memory)
        // Using newBufferWithBytesNoCopy to avoid copying
        // Note: deallocator:nil means the buffer does not own the memory,
        // so we rely on @autoreleasepool for automatic release
        id<MTLBuffer> buffer_a = [device newBufferWithBytesNoCopy:(void*)a
                                                           length:size * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        id<MTLBuffer> buffer_b = [device newBufferWithBytesNoCopy:(void*)b
                                                           length:size * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        id<MTLBuffer> buffer_c = [device newBufferWithBytesNoCopy:c
                                                           length:size * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];

        dispatchElementwiseKernel(impl_->add_pipeline,
                                  impl_->command_queue,
                                  buffer_a,
                                  buffer_b,
                                  buffer_c,
                                  static_cast<uint32_t>(size));

        // No manual release: @autoreleasepool handles it automatically
    }
}

void MetalKernels::mul(const float* a, const float* b, float* c, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = impl_->device;

        id<MTLBuffer> buffer_a = [device newBufferWithBytesNoCopy:(void*)a
                                                           length:size * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        id<MTLBuffer> buffer_b = [device newBufferWithBytesNoCopy:(void*)b
                                                           length:size * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        id<MTLBuffer> buffer_c = [device newBufferWithBytesNoCopy:c
                                                           length:size * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];

        dispatchElementwiseKernel(impl_->mul_pipeline,
                                  impl_->command_queue,
                                  buffer_a,
                                  buffer_b,
                                  buffer_c,
                                  static_cast<uint32_t>(size));

        // No manual release: @autoreleasepool handles it automatically
    }
}

void MetalKernels::sub(const float* a, const float* b, float* c, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = impl_->device;

        id<MTLBuffer> buffer_a = [device newBufferWithBytesNoCopy:(void*)a
                                                           length:size * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        id<MTLBuffer> buffer_b = [device newBufferWithBytesNoCopy:(void*)b
                                                           length:size * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        id<MTLBuffer> buffer_c = [device newBufferWithBytesNoCopy:c
                                                           length:size * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];

        dispatchElementwiseKernel(impl_->sub_pipeline,
                                  impl_->command_queue,
                                  buffer_a,
                                  buffer_b,
                                  buffer_c,
                                  static_cast<uint32_t>(size));

        // No manual release: @autoreleasepool handles it automatically
    }
}

void MetalKernels::div(const float* a, const float* b, float* c, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = impl_->device;

        id<MTLBuffer> buffer_a = [device newBufferWithBytesNoCopy:(void*)a
                                                           length:size * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        id<MTLBuffer> buffer_b = [device newBufferWithBytesNoCopy:(void*)b
                                                           length:size * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        id<MTLBuffer> buffer_c = [device newBufferWithBytesNoCopy:c
                                                           length:size * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];

        dispatchElementwiseKernel(impl_->div_pipeline,
                                  impl_->command_queue,
                                  buffer_a,
                                  buffer_b,
                                  buffer_c,
                                  static_cast<uint32_t>(size));

        // No manual release: @autoreleasepool handles it automatically
    }
}

void MetalKernels::relu(const float* x, float* y, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = impl_->device;
        id<MTLCommandQueue> queue = impl_->command_queue;

        id<MTLBuffer> buffer_x = [device newBufferWithBytesNoCopy:(void*)x
                                                           length:size * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        id<MTLBuffer> buffer_y = [device newBufferWithBytesNoCopy:y
                                                           length:size * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:impl_->relu_pipeline];
        [encoder setBuffer:buffer_x offset:0 atIndex:0];
        [encoder setBuffer:buffer_y offset:0 atIndex:1];
        uint32_t size_u32 = static_cast<uint32_t>(size);
        [encoder setBytes:&size_u32 length:sizeof(uint32_t) atIndex:2];

        // Thread group size: 256
        NSUInteger threadGroupSize = 256;
        NSUInteger numThreadGroups = (size + threadGroupSize - 1) / threadGroupSize;

        MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize numGroups = MTLSizeMake(numThreadGroups, 1, 1);

        [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // No manual release: @autoreleasepool handles it automatically
    }
}

void MetalKernels::sum(const float* input, float* output, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = impl_->device;
        id<MTLCommandQueue> queue = impl_->command_queue;

        // Thread group size
        NSUInteger threadGroupSize = 256;
        NSUInteger numGroups = (size + threadGroupSize - 1) / threadGroupSize;

        // Create buffers
        id<MTLBuffer> input_buffer = [device newBufferWithBytesNoCopy:(void*)input
                                                               length:size * sizeof(float)
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
        id<MTLBuffer> partial_buffer = [device newBufferWithLength:numGroups * sizeof(float)
                                                           options:MTLResourceStorageModeShared];

        // ===== Stage 1: Partial sums =====
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:impl_->sum_stage1_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:partial_buffer offset:0 atIndex:1];
        uint32_t size_uint = static_cast<uint32_t>(size);
        [encoder setBytes:&size_uint length:sizeof(uint32_t) atIndex:2];
        [encoder setThreadgroupMemoryLength:threadGroupSize * sizeof(float) atIndex:0];

        MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize groups = MTLSizeMake(numGroups, 1, 1);
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // ===== Stage 2: Final sum =====
        id<MTLBuffer> output_buffer = [device newBufferWithBytesNoCopy:output
                                                                length:sizeof(float)
                                                               options:MTLResourceStorageModeShared
                                                           deallocator:nil];

        commandBuffer = [queue commandBuffer];
        encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:impl_->sum_stage2_pipeline];
        [encoder setBuffer:partial_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        uint32_t num_partials = static_cast<uint32_t>(numGroups);
        [encoder setBytes:&num_partials length:sizeof(uint32_t) atIndex:2];
        [encoder setThreadgroupMemoryLength:threadGroupSize * sizeof(float) atIndex:0];

        MTLSize finalGroup = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize oneGroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreadgroups:oneGroup threadsPerThreadgroup:finalGroup];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // No manual release for buffers created with deallocator:nil
        // partial_buffer is released automatically by ARC
    }
}

void MetalKernels::mean(const float* input, float* output, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = impl_->device;
        id<MTLCommandQueue> queue = impl_->command_queue;

        // First compute sum
        id<MTLBuffer> temp_sum_buffer = [device newBufferWithLength:sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        float* temp_sum = (float*)[temp_sum_buffer contents];

        sum(input, temp_sum, size);

        // Then divide by size
        id<MTLBuffer> sum_buffer = [device newBufferWithBytesNoCopy:temp_sum
                                                             length:sizeof(float)
                                                            options:MTLResourceStorageModeShared
                                                        deallocator:nil];
        id<MTLBuffer> output_buffer = [device newBufferWithBytesNoCopy:output
                                                                length:sizeof(float)
                                                               options:MTLResourceStorageModeShared
                                                           deallocator:nil];

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:impl_->mean_pipeline];
        [encoder setBuffer:sum_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        uint32_t size_uint = static_cast<uint32_t>(size);
        [encoder setBytes:&size_uint length:sizeof(uint32_t) atIndex:2];

        MTLSize threadsPerGroup = MTLSizeMake(1, 1, 1);
        MTLSize groups = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // No manual release for buffers created with deallocator:nil
        // temp_sum_buffer is released automatically by ARC
    }
}

void MetalKernels::matmul(const float* a, const float* b, float* c, size_t m, size_t k, size_t n) {
    @autoreleasepool {
        id<MTLDevice> device = impl_->device;
        id<MTLCommandQueue> queue = impl_->command_queue;

        // Wrap buffers
        id<MTLBuffer> buffer_a = [device newBufferWithBytesNoCopy:(void*)a
                                                           length:m * k * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        id<MTLBuffer> buffer_b = [device newBufferWithBytesNoCopy:(void*)b
                                                           length:k * n * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        id<MTLBuffer> buffer_c = [device newBufferWithBytesNoCopy:c
                                                           length:m * n * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];

        // Create MPS matrix descriptors
        MPSMatrixDescriptor* desc_a =
            [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                  columns:k
                                                 rowBytes:k * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* desc_b =
            [MPSMatrixDescriptor matrixDescriptorWithRows:k
                                                  columns:n
                                                 rowBytes:n * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* desc_c =
            [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                  columns:n
                                                 rowBytes:n * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];

        // Create MPS matrices
        MPSMatrix* matrix_a = [[MPSMatrix alloc] initWithBuffer:buffer_a descriptor:desc_a];
        MPSMatrix* matrix_b = [[MPSMatrix alloc] initWithBuffer:buffer_b descriptor:desc_b];
        MPSMatrix* matrix_c = [[MPSMatrix alloc] initWithBuffer:buffer_c descriptor:desc_c];

        // Create MPS matrix multiplication
        MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                            transposeLeft:NO
                                                                           transposeRight:NO
                                                                               resultRows:m
                                                                            resultColumns:n
                                                                          interiorColumns:k
                                                                                    alpha:1.0
                                                                                     beta:0.0];

        // Execute
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        [matmul encodeToCommandBuffer:commandBuffer
                           leftMatrix:matrix_a
                          rightMatrix:matrix_b
                         resultMatrix:matrix_c];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Only release MPS objects (matrices and matmul)
        // Buffers created with deallocator:nil are released automatically by @autoreleasepool
        [matmul release];
        [matrix_c release];
        [matrix_b release];
        [matrix_a release];
    }
}

void MetalKernels::synchronize() {
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [impl_->command_queue commandBuffer];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

}  // namespace gpu
}  // namespace gradflow
