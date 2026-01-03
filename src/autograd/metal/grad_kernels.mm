// src/autograd/metal/grad_kernels.mm
// Metal Gradient Compute Shaders C++ Wrapper Implementation
// Copyright (c) 2025 GradFlow Project

#include "gradflow/autograd/metal/grad_kernels.hpp"

#include "gradflow/autograd/metal/device.hpp"
#include "gradflow/autograd/metal/kernels.hpp"

#import <Metal/Metal.h>

#include <cstring>
#include <stdexcept>
#include <string>

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace gradflow {
namespace gpu {

// ===================================================================
// MetalGradKernelsImpl: Internal implementation class
// ===================================================================

class MetalGradKernelsImpl {
public:
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    id<MTLLibrary> library;

    // Compute Pipeline States for gradient kernels
    id<MTLComputePipelineState> mul_grad_x_pipeline;
    id<MTLComputePipelineState> mul_grad_y_pipeline;
    id<MTLComputePipelineState> relu_grad_pipeline;

    // Reuse existing forward kernels for matmul gradients
    MetalKernels* forward_kernels;

    MetalGradKernelsImpl(MetalDevice* metal_device) {
        device = (id<MTLDevice>)metal_device->getMetalDevice();
        command_queue = (id<MTLCommandQueue>)metal_device->getMetalCommandQueue();

        // Create MetalKernels instance for reusing matmul
        forward_kernels = new MetalKernels(metal_device);

        NSError* error = nil;

        // 優先順位 1: 事前コンパイルされた .metallib をロード（デプロイメント向け）
#ifdef GRADFLOW_METAL_GRAD_LIB_PATH
        NSString* metallibPath = @GRADFLOW_METAL_GRAD_LIB_PATH;
        library = [device newLibraryWithFile:metallibPath error:&error];

        if (!library) {
            error = nil;  // エラーをリセットして次のフォールバックへ
        }
#endif

        // 優先順位 2: ソースディレクトリの .metal ファイルをロード（開発環境向け）
        if (!library) {
            const char* source_path = __FILE__;  // grad_kernels.mm path
            std::string source_dir(source_path);
            size_t last_slash = source_dir.find_last_of("/\\");
            if (last_slash != std::string::npos) {
                source_dir = source_dir.substr(0, last_slash);
            }
            std::string metal_source_path = source_dir + "/grad_kernels.metal";

            NSString* sourcePath = [NSString stringWithUTF8String:metal_source_path.c_str()];
            NSString* sourceCode = [NSString stringWithContentsOfFile:sourcePath
                                                             encoding:NSUTF8StringEncoding
                                                                error:&error];

            if (sourceCode) {
                // Runtime コンパイル
                library = [device newLibraryWithSource:sourceCode options:nil error:&error];
            }
        }

        // すべてのフォールバックが失敗
        if (!library) {
            std::string error_msg = "Failed to load Metal gradient library. Tried:\n";
#ifdef GRADFLOW_METAL_GRAD_LIB_PATH
            error_msg += "1. Precompiled .metallib: " GRADFLOW_METAL_GRAD_LIB_PATH "\n";
#endif
            error_msg += "2. Runtime compile from source (grad_kernels.metal)\n";
            if (error) {
                error_msg +=
                    "Last error: " + std::string([[error localizedDescription] UTF8String]);
            }
            throw std::runtime_error(error_msg);
        }

        // Create compute pipeline states
        mul_grad_x_pipeline = createPipeline("mul_grad_x_kernel");
        mul_grad_y_pipeline = createPipeline("mul_grad_y_kernel");
        relu_grad_pipeline = createPipeline("relu_grad_kernel");
    }

    ~MetalGradKernelsImpl() { delete forward_kernels; }

    id<MTLComputePipelineState> createPipeline(const char* kernel_name) {
        NSString* kernelName = [NSString stringWithUTF8String:kernel_name];
        id<MTLFunction> function = [library newFunctionWithName:kernelName];

        if (!function) {
            std::string error_msg = "Failed to find Metal function: ";
            error_msg += kernel_name;
            throw std::runtime_error(error_msg);
        }

        NSError* error = nil;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                     error:&error];

        if (!pipeline) {
            std::string error_msg = "Failed to create compute pipeline for: ";
            error_msg += kernel_name;
            if (error) {
                error_msg += " Error: " + std::string([[error localizedDescription] UTF8String]);
            }
            throw std::runtime_error(error_msg);
        }

        return pipeline;
    }
};

// ===================================================================
// MetalGradKernels: Public interface implementation
// ===================================================================

MetalGradKernels::MetalGradKernels(MetalDevice* device)
    : impl_(std::make_unique<MetalGradKernelsImpl>(device)) {}

MetalGradKernels::~MetalGradKernels() = default;

void MetalGradKernels::add_grad(const float* grad_output,
                                float* grad_x,
                                float* grad_y,
                                size_t size) {
    // Add の勾配は恒等写像なので、CPU の memcpy で実装
    //
    // 実装方針:
    // - Unified Memory 環境では CPU からの memcpy は高速に動作する
    // - GPU カーネルの起動オーバーヘッドが演算時間を上回る可能性が高い
    // - 単純なメモリコピーであり、GPU 並列化のメリットが小さい
    //
    // 将来の最適化オプション:
    // - 大規模テンソル (> 1MB) で CPU memcpy がボトルネックになる場合、
    //   MTLBlitCommandEncoder を使用した GPU コピーへの置き換えを検討
    // - ベンチマーク結果に基づいて実装を選択する
    std::memcpy(grad_x, grad_output, size * sizeof(float));
    std::memcpy(grad_y, grad_output, size * sizeof(float));
}

void MetalGradKernels::mul_grad(const float* grad_output,
                                const float* x,
                                const float* y,
                                float* grad_x,
                                float* grad_y,
                                size_t size) {
    // 空テンソル (size = 0) の場合は何もしない
    if (size == 0) {
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = impl_->device;
        id<MTLCommandQueue> queue = impl_->command_queue;

        id<MTLBuffer> buffer_grad_output =
            [device newBufferWithBytesNoCopy:(void*)grad_output
                                      length:size * sizeof(float)
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
        id<MTLBuffer> buffer_x = [device newBufferWithBytesNoCopy:(void*)x
                                                           length:size * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        id<MTLBuffer> buffer_y = [device newBufferWithBytesNoCopy:(void*)y
                                                           length:size * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        id<MTLBuffer> buffer_grad_x = [device newBufferWithBytesNoCopy:grad_x
                                                                length:size * sizeof(float)
                                                               options:MTLResourceStorageModeShared
                                                           deallocator:nil];
        id<MTLBuffer> buffer_grad_y = [device newBufferWithBytesNoCopy:grad_y
                                                                length:size * sizeof(float)
                                                               options:MTLResourceStorageModeShared
                                                           deallocator:nil];

        // Check for allocation failures
        if (!buffer_grad_output || !buffer_x || !buffer_y || !buffer_grad_x || !buffer_grad_y) {
            throw std::runtime_error(
                "Failed to allocate Metal buffers for mul_grad. "
                "GPU memory may be insufficient.");
        }

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> compute_encoder = [command_buffer computeCommandEncoder];

        // Thread group size
        // 注: 256 はバランスの取れた値として選択
        // - Apple Silicon の推奨範囲: 256-1024
        // - 小さすぎると GPU の並列度を活用できない
        // - 大きすぎると小規模テンソルで無駄なスレッドが発生
        // - 現在の実装では 256 で十分な性能が得られているため、
        //   パフォーマンス問題が確認されるまで変更しない
        NSUInteger thread_group_size = impl_->mul_grad_x_pipeline.maxTotalThreadsPerThreadgroup;
        if (thread_group_size > 256) {
            thread_group_size = 256;
        }
        MTLSize threads_per_group = MTLSizeMake(thread_group_size, 1, 1);
        MTLSize num_thread_groups =
            MTLSizeMake((size + thread_group_size - 1) / thread_group_size, 1, 1);

        uint32_t size_u32 = static_cast<uint32_t>(size);

        // grad_x = grad_output * y
        [compute_encoder setComputePipelineState:impl_->mul_grad_x_pipeline];
        [compute_encoder setBuffer:buffer_grad_output offset:0 atIndex:0];
        [compute_encoder setBuffer:buffer_y offset:0 atIndex:1];
        [compute_encoder setBuffer:buffer_grad_x offset:0 atIndex:2];
        [compute_encoder setBytes:&size_u32 length:sizeof(uint32_t) atIndex:3];
        [compute_encoder dispatchThreadgroups:num_thread_groups
                        threadsPerThreadgroup:threads_per_group];

        // grad_y = grad_output * x
        [compute_encoder setComputePipelineState:impl_->mul_grad_y_pipeline];
        [compute_encoder setBuffer:buffer_grad_output offset:0 atIndex:0];
        [compute_encoder setBuffer:buffer_x offset:0 atIndex:1];
        [compute_encoder setBuffer:buffer_grad_y offset:0 atIndex:2];
        [compute_encoder setBytes:&size_u32 length:sizeof(uint32_t) atIndex:3];
        [compute_encoder dispatchThreadgroups:num_thread_groups
                        threadsPerThreadgroup:threads_per_group];

        [compute_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }
}

void MetalGradKernels::relu_grad(const float* grad_output,
                                 const float* x,
                                 float* grad_x,
                                 size_t size) {
    // 空テンソル (size = 0) の場合は何もしない
    if (size == 0) {
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = impl_->device;
        id<MTLCommandQueue> queue = impl_->command_queue;

        id<MTLBuffer> buffer_grad_output =
            [device newBufferWithBytesNoCopy:(void*)grad_output
                                      length:size * sizeof(float)
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
        id<MTLBuffer> buffer_x = [device newBufferWithBytesNoCopy:(void*)x
                                                           length:size * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        id<MTLBuffer> buffer_grad_x = [device newBufferWithBytesNoCopy:grad_x
                                                                length:size * sizeof(float)
                                                               options:MTLResourceStorageModeShared
                                                           deallocator:nil];

        // Check for allocation failures
        if (!buffer_grad_output || !buffer_x || !buffer_grad_x) {
            throw std::runtime_error(
                "Failed to allocate Metal buffers for relu_grad. "
                "GPU memory may be insufficient.");
        }

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> compute_encoder = [command_buffer computeCommandEncoder];

        // Thread group size
        // 注: 256 はバランスの取れた値として選択（詳細は mul_grad を参照）
        NSUInteger thread_group_size = impl_->relu_grad_pipeline.maxTotalThreadsPerThreadgroup;
        if (thread_group_size > 256) {
            thread_group_size = 256;
        }
        MTLSize threads_per_group = MTLSizeMake(thread_group_size, 1, 1);
        MTLSize num_thread_groups =
            MTLSizeMake((size + thread_group_size - 1) / thread_group_size, 1, 1);

        uint32_t size_u32 = static_cast<uint32_t>(size);

        [compute_encoder setComputePipelineState:impl_->relu_grad_pipeline];
        [compute_encoder setBuffer:buffer_grad_output offset:0 atIndex:0];
        [compute_encoder setBuffer:buffer_x offset:0 atIndex:1];
        [compute_encoder setBuffer:buffer_grad_x offset:0 atIndex:2];
        [compute_encoder setBytes:&size_u32 length:sizeof(uint32_t) atIndex:3];
        [compute_encoder dispatchThreadgroups:num_thread_groups
                        threadsPerThreadgroup:threads_per_group];

        [compute_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }
}

void MetalGradKernels::matmul_grad_x(
    const float* grad_output, const float* y_t, float* grad_x, size_t m, size_t k, size_t n) {
    // grad_x = grad_output @ y^T
    // grad_output: (m, n), y_t: (n, k) -> grad_x: (m, k)
    // 既存の matmul カーネルを再利用
    impl_->forward_kernels->matmul(grad_output, y_t, grad_x, m, n, k);
}

void MetalGradKernels::matmul_grad_y(
    const float* x_t, const float* grad_output, float* grad_y, size_t k, size_t m, size_t n) {
    // grad_y = x^T @ grad_output
    // x_t: (k, m), grad_output: (m, n) -> grad_y: (k, n)
    // 既存の matmul カーネルを再利用
    impl_->forward_kernels->matmul(x_t, grad_output, grad_y, k, m, n);
}

void MetalGradKernels::synchronize() {
    // Command queue が空になるまで待機
    // Note: 各操作は既に waitUntilCompleted を呼んでいるので、
    // この関数は明示的な同期のために提供されています
    id<MTLCommandBuffer> command_buffer = [impl_->command_queue commandBuffer];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

}  // namespace gpu
}  // namespace gradflow
