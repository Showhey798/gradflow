# Issue #15: Metal Compute Shader の詳細設計書

## 1. 調査・リサーチ結果

### Metal Compute Shader のベストプラクティス

Apple Silicon の Metal GPU は、TBDR (Tile-Based Deferred Rendering) アーキテクチャを採用し、Unified Memory Architecture により CPU と GPU が同一のメモリプールを共有します。これにより、従来の CUDA のような明示的なメモリコピーのオーバーヘッドを削減できます。

#### 重要な最適化原則

**1. Thread Group Size の最適化**

- **SIMD Group Barriers**: Thread group が単一の SIMD (32 threads) に収まる場合、通常の thread group barrier 関数は不要です。代わりに SIMD group barrier 関数を使用することで、より高速な同期が可能になります。
- **Max-Total-Threads-Per-Threadgroup のチューニング**: この値を調整することで、ターゲット occupancy を変更し、パフォーマンスのスイートスポットを見つけることができます。
- **推奨サイズ**: Apple Silicon では、32, 64, 128, 256 スレッドが一般的に使用されます。特に 256 スレッドは多くのケースで良好なパフォーマンスを示します。

**2. Compute Thread の効率化**

- **ワークアイテムの再利用**: 単一の compute thread で複数の概念的なワークアイテムを処理し、値を thread group memory を通さずに再利用するパターンが推奨されます。
- **Thread Occupancy の向上**: レイテンシー隠蔽を改善するため、thread occupancy を高めることが重要です。レジスタやメモリの最適化使用により occupancy を向上できます。

**3. メモリアクセスの最適化**

- **Threadgroup Memory の使用判断**: 最新の Apple Family 9 GPU では、threadgroup memory が高速化されていますが、texture cache のミス率が低い場合はレジスタを直接使用する方が効率的です。
- **Unified Memory の活用**: MTLResourceStorageModeShared を使用することで、CPU と GPU 間のデータ転送オーバーヘッドを最小化できます。

### Metal Performance Shaders (MPS) の統合

**MPSMatrixMultiplication の C++ からの利用**

Metal API は 2022 年の WWDC で C++17 サポートが発表され、metal-cpp バインディングが提供されています。しかし、MPSMatrixMultiplication は現在 Swift と Objective-C でのみ利用可能であり、C++ から直接呼び出すには Objective-C++ ブリッジが必要です。

**パフォーマンス特性**:
- 十分に大きな行列サイズで約 7 TFLOPS のパフォーマンスを達成
- 小さい行列では BLAS が高速、大きい行列では MPS が高速

### Metal Shading Language (MSL) のカーネル実装

**Elementwise 演算**:
- MSL ではベクトルに対して直接算術演算を適用可能（例: `float4 sum = c1 + c2`）
- Kernel 関数は 1D, 2D, 3D グリッド上で実行されるデータ並列関数

**Reduction 演算**:
- Thread group memory を使用した並列 reduction パターン
- Atomic operations による集約（`atomic_fetch_add_explicit`）
- 2 段階 reduction: local reduction (thread group) + global reduction

### 参考文献

- [Learn performance best practices for Metal shaders - Tech Talks](https://developer.apple.com/videos/play/tech-talks/111373/) - Metal シェーダーの最新パフォーマンスベストプラクティス
- [Optimize Metal Performance for Apple silicon Macs - WWDC20](https://developer.apple.com/videos/play/wwdc2020/10632/) - Apple Silicon 向け最適化
- [Calculating threadgroup and grid sizes | Apple Developer Documentation](https://developer.apple.com/documentation/metal/compute_passes/calculating_threadgroup_and_grid_sizes) - Thread group サイズの計算方法
- [Metal Shading Language Specification Version 4](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) - MSL 公式仕様書 (2025 年 10 月更新)
- [MPSMatrixMultiplication | Apple Developer Documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication) - MPS 行列乗算
- [GitHub - bkvogel/metal_performance_testing](https://github.com/bkvogel/metal_performance_testing) - Metal による行列乗算の C++ 実装例
- [Matrix Multiplication with Metal Performance Shaders](https://machinethink.net/blog/mps-matrix-multiplication/) - MPS 行列乗算の詳細解説

---

## 2. 分析と評価

### 現状の課題

- Issue #14 で MetalDevice と MetalAllocator が実装済み（`gradflow::gpu` 名前空間）
- GPU メモリの割り当ては可能だが、実際の演算カーネルが未実装
- CPU のみでの計算では、大規模テンソル演算のパフォーマンスが限定的
- Apple Silicon の GPU 能力を活用できていない

### 採用すべき設計原則

#### 1. 単一責任の原則 (SRP)

- **MetalKernels**: Metal Compute Shader カーネルの管理のみを担当
- **MetalOperations**: C++ から Metal カーネルを呼び出す高レベル API を提供
- 各カーネル（add, mul, sum, mean, matmul）は独立した関数として実装

#### 2. 開放閉鎖原則 (OCP)

- 新しいカーネル（例: convolution, pooling）を追加する際、既存コードを変更せずに拡張可能
- MSL ファイルと C++ ラッパーを追加するだけで新規演算を統合

#### 3. 依存性逆転の原則 (DIP)

- 高レベルモジュール（Tensor, Operation）は Metal 固有の詳細に依存せず、抽象インターフェースを通じて利用
- Metal Shading Language の詳細は .metal ファイルと .mm ファイル内に隠蔽

#### 4. RAII (Resource Acquisition Is Initialization)

- MTLComputePipelineState, MTLCommandBuffer のライフタイムは C++ オブジェクトのスコープと連動
- デストラクタで自動的に Metal リソースを解放

#### 5. パフォーマンス最適化

- **Thread Group Size**: 256 スレッドを基本とし、演算ごとに最適化
- **Occupancy**: レジスタ使用量を抑え、高い occupancy を維持
- **Memory Access Pattern**: Coalesced access を実現し、メモリバンド幅を最大化

---

## 3. 推奨アーキテクチャ案

### 設計のコンセプト

**3 層アーキテクチャ**:

1. **Metal Shading Language Layer** (`src/autograd/metal/kernels.metal`): GPU カーネルの実装
2. **C++ Wrapper Layer** (`src/autograd/metal/kernels.mm`): Metal API の呼び出しと C++ インターフェース
3. **Public Interface Layer** (`include/gradflow/autograd/metal/kernels.hpp`): クリーンな C++ API

**主要コンポーネント**:

- **MetalKernels**: すべての Metal カーネルを管理するクラス
- **Kernel 関数群**: add, mul, sub, div, sum, mean, matmul など
- **MPS 統合**: MPSMatrixMultiplication による高速行列乗算

### クラス設計

#### 3.1 MetalKernels クラス（C++ Public Interface）

```cpp
// include/gradflow/autograd/metal/kernels.hpp
#pragma once

#ifdef __APPLE__

#include <cstddef>
#include <memory>
#include <string>

namespace gradflow {
namespace gpu {

// Forward declaration
class MetalDevice;
class MetalKernelsImpl;

/**
 * @brief Metal Compute Shader カーネルの管理クラス
 *
 * Metal GPU 上での各種演算カーネルを提供します。
 * - Elementwise 演算: add, mul, sub, div
 * - Reduction 演算: sum, mean
 * - 行列演算: matmul (MPS 統合)
 *
 * 使用例:
 * @code
 *   auto device = MetalDevice::create();
 *   MetalKernels kernels(device.get());
 *
 *   float* a = allocator.allocate(1024 * sizeof(float));
 *   float* b = allocator.allocate(1024 * sizeof(float));
 *   float* c = allocator.allocate(1024 * sizeof(float));
 *
 *   kernels.add(a, b, c, 1024);  // c = a + b
 * @endcode
 */
class MetalKernels {
 public:
  /**
   * @brief MetalKernels を構築
   *
   * @param device Metal デバイス (非 null)
   * @throws std::runtime_error カーネルのロード失敗時
   */
  explicit MetalKernels(MetalDevice* device);

  ~MetalKernels();

  // コピー・ムーブ禁止
  MetalKernels(const MetalKernels&) = delete;
  MetalKernels& operator=(const MetalKernels&) = delete;
  MetalKernels(MetalKernels&&) = delete;
  MetalKernels& operator=(MetalKernels&&) = delete;

  // ===== Elementwise Operations =====

  /**
   * @brief 要素ごとの加算: c = a + b
   *
   * @param a 入力配列 A (GPU メモリ)
   * @param b 入力配列 B (GPU メモリ)
   * @param c 出力配列 C (GPU メモリ)
   * @param size 要素数
   */
  void add(const float* a, const float* b, float* c, size_t size);

  /**
   * @brief 要素ごとの乗算: c = a * b
   *
   * @param a 入力配列 A (GPU メモリ)
   * @param b 入力配列 B (GPU メモリ)
   * @param c 出力配列 C (GPU メモリ)
   * @param size 要素数
   */
  void mul(const float* a, const float* b, float* c, size_t size);

  /**
   * @brief 要素ごとの減算: c = a - b
   *
   * @param a 入力配列 A (GPU メモリ)
   * @param b 入力配列 B (GPU メモリ)
   * @param c 出力配列 C (GPU メモリ)
   * @param size 要素数
   */
  void sub(const float* a, const float* b, float* c, size_t size);

  /**
   * @brief 要素ごとの除算: c = a / b
   *
   * @param a 入力配列 A (GPU メモリ)
   * @param b 入力配列 B (GPU メモリ)
   * @param c 出力配列 C (GPU メモリ)
   * @param size 要素数
   */
  void div(const float* a, const float* b, float* c, size_t size);

  // ===== Reduction Operations =====

  /**
   * @brief 総和を計算: result = sum(input)
   *
   * @param input 入力配列 (GPU メモリ)
   * @param output 出力スカラー (GPU メモリ, 1 要素)
   * @param size 要素数
   */
  void sum(const float* input, float* output, size_t size);

  /**
   * @brief 平均値を計算: result = mean(input)
   *
   * @param input 入力配列 (GPU メモリ)
   * @param output 出力スカラー (GPU メモリ, 1 要素)
   * @param size 要素数
   */
  void mean(const float* input, float* output, size_t size);

  // ===== Matrix Operations =====

  /**
   * @brief 行列乗算: C = A * B (MPS 使用)
   *
   * Metal Performance Shaders の MPSMatrixMultiplication を使用します。
   *
   * @param a 入力行列 A (GPU メモリ, row-major)
   * @param b 入力行列 B (GPU メモリ, row-major)
   * @param c 出力行列 C (GPU メモリ, row-major)
   * @param m 行列 A の行数
   * @param k 行列 A の列数 / 行列 B の行数
   * @param n 行列 B の列数
   */
  void matmul(const float* a, const float* b, float* c, size_t m, size_t k,
              size_t n);

  /**
   * @brief GPU 操作を同期
   *
   * 保留中のすべてのコマンドが完了するまで待機します。
   */
  void synchronize();

 private:
  std::unique_ptr<MetalKernelsImpl> impl_;
};

}  // namespace gpu
}  // namespace gradflow

#endif  // __APPLE__
```

---

## 4. 実装の詳細

### 4.1 Metal Shading Language カーネル (kernels.metal)

```metal
// src/autograd/metal/kernels.metal
#include <metal_stdlib>
using namespace metal;

// ===== Elementwise Operations =====

/**
 * @brief 要素ごとの加算カーネル: c = a + b
 *
 * Thread Group Size: 256
 * Grid Size: (size + 255) / 256
 */
kernel void add_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        c[gid] = a[gid] + b[gid];
    }
}

/**
 * @brief 要素ごとの乗算カーネル: c = a * b
 */
kernel void mul_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        c[gid] = a[gid] * b[gid];
    }
}

/**
 * @brief 要素ごとの減算カーネル: c = a - b
 */
kernel void sub_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        c[gid] = a[gid] - b[gid];
    }
}

/**
 * @brief 要素ごとの除算カーネル: c = a / b
 */
kernel void div_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        c[gid] = a[gid] / b[gid];
    }
}

// ===== Reduction Operations =====

/**
 * @brief 総和カーネル (2段階reduction)
 *
 * Stage 1: Thread group ごとに local sum を計算
 * Thread Group Size: 256
 * Threadgroup Memory: 256 * sizeof(float)
 */
kernel void sum_kernel_stage1(
    device const float* input [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    threadgroup float* local_sum [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]]
) {
    // Load data into threadgroup memory
    local_sum[tid] = (gid < size) ? input[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction in threadgroup
    for (uint stride = tpg / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            local_sum[tid] += local_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partial sum
    if (tid == 0) {
        uint group_id = gid / tpg;
        partial_sums[group_id] = local_sum[0];
    }
}

/**
 * @brief 総和カーネル (Stage 2: partial sums を集約)
 *
 * Thread Group Size: 256
 */
kernel void sum_kernel_stage2(
    device const float* partial_sums [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& num_partials [[buffer(2)]],
    threadgroup float* local_sum [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    // Load partial sums
    local_sum[tid] = (tid < num_partials) ? partial_sums[tid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = tpg / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            local_sum[tid] += local_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write final sum
    if (tid == 0) {
        output[0] = local_sum[0];
    }
}

/**
 * @brief 平均値カーネル (sum を size で割る)
 */
kernel void mean_kernel(
    device const float* sum [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        output[0] = sum[0] / static_cast<float>(size);
    }
}
```

### 4.2 C++ ラッパー (kernels.mm)

```cpp
// src/autograd/metal/kernels.mm
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "gradflow/autograd/metal/kernels.hpp"
#include "gradflow/autograd/metal/device.hpp"
#include <stdexcept>
#include <string>

namespace gradflow {
namespace gpu {

// 内部実装
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
  id<MTLComputePipelineState> sum_stage1_pipeline;
  id<MTLComputePipelineState> sum_stage2_pipeline;
  id<MTLComputePipelineState> mean_pipeline;

  MetalKernelsImpl(MetalDevice* metal_device) {
    device = (id<MTLDevice>)metal_device->getMetalDevice();
    command_queue = (id<MTLCommandQueue>)metal_device->getMetalCommandQueue();

    // Load Metal library
    NSError* error = nil;
    NSString* libraryPath = [[NSBundle mainBundle]
        pathForResource:@"kernels" ofType:@"metallib"];

    if (libraryPath) {
      library = [device newLibraryWithFile:libraryPath error:&error];
    } else {
      // Fallback: load default library (for development)
      library = [device newDefaultLibrary];
    }

    if (!library) {
      throw std::runtime_error(
          "Failed to load Metal library: " +
          std::string([[error localizedDescription] UTF8String]));
    }

    // Create compute pipeline states
    add_pipeline = createPipeline("add_kernel");
    mul_pipeline = createPipeline("mul_kernel");
    sub_pipeline = createPipeline("sub_kernel");
    div_pipeline = createPipeline("div_kernel");
    sum_stage1_pipeline = createPipeline("sum_kernel_stage1");
    sum_stage2_pipeline = createPipeline("sum_kernel_stage2");
    mean_pipeline = createPipeline("mean_kernel");
  }

  ~MetalKernelsImpl() {
    [mean_pipeline release];
    [sum_stage2_pipeline release];
    [sum_stage1_pipeline release];
    [div_pipeline release];
    [sub_pipeline release];
    [mul_pipeline release];
    [add_pipeline release];
    [library release];
  }

 private:
  id<MTLComputePipelineState> createPipeline(const char* function_name) {
    NSError* error = nil;
    id<MTLFunction> function =
        [library newFunctionWithName:@(function_name)];
    if (!function) {
      throw std::runtime_error(std::string("Failed to find function: ") +
                               function_name);
    }

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    [function release];

    if (!pipeline) {
      throw std::runtime_error(
          std::string("Failed to create pipeline for ") + function_name +
          ": " + std::string([[error localizedDescription] UTF8String]));
    }

    return pipeline;
  }
};

// MetalKernels の実装

MetalKernels::MetalKernels(MetalDevice* device)
    : impl_(std::make_unique<MetalKernelsImpl>(device)) {}

MetalKernels::~MetalKernels() = default;

// Helper: Elementwise カーネル実行
static void dispatchElementwiseKernel(id<MTLComputePipelineState> pipeline,
                                      id<MTLCommandQueue> queue,
                                      id<MTLBuffer> buffer_a,
                                      id<MTLBuffer> buffer_b,
                                      id<MTLBuffer> buffer_c,
                                      uint32_t size) {
  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];

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

    [encoder dispatchThreadgroups:numGroups
            threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
  }
}

// Helper: MTLBuffer を取得 (Unified Memory)
static id<MTLBuffer> getBufferFromPointer(const void* ptr,
                                          MetalAllocator* allocator) {
  // MetalAllocator が管理している MTLBuffer を取得
  // 実装の詳細は allocator.mm を参照
  // ここでは簡易的に実装を示します
  return nil;  // 実際の実装では allocator の内部データから取得
}

void MetalKernels::add(const float* a, const float* b, float* c, size_t size) {
  // MTLBuffer の取得 (Unified Memory なのでポインタから逆引き)
  // 実装の詳細: MetalAllocator に buffer_map_ があるため、そこから取得
  // ここでは簡略化のため、直接 MTLBuffer を作成
  @autoreleasepool {
    id<MTLDevice> device = impl_->device;

    // Wrap existing memory as MTLBuffer (Unified Memory)
    id<MTLBuffer> buffer_a = [device
        newBufferWithBytesNoCopy:(void*)a
                          length:size * sizeof(float)
                         options:MTLResourceStorageModeShared
                     deallocator:nil];
    id<MTLBuffer> buffer_b = [device
        newBufferWithBytesNoCopy:(void*)b
                          length:size * sizeof(float)
                         options:MTLResourceStorageModeShared
                     deallocator:nil];
    id<MTLBuffer> buffer_c = [device
        newBufferWithBytesNoCopy:c
                          length:size * sizeof(float)
                         options:MTLResourceStorageModeShared
                     deallocator:nil];

    dispatchElementwiseKernel(impl_->add_pipeline, impl_->command_queue,
                              buffer_a, buffer_b, buffer_c,
                              static_cast<uint32_t>(size));

    [buffer_c release];
    [buffer_b release];
    [buffer_a release];
  }
}

void MetalKernels::mul(const float* a, const float* b, float* c, size_t size) {
  @autoreleasepool {
    id<MTLDevice> device = impl_->device;

    id<MTLBuffer> buffer_a = [device
        newBufferWithBytesNoCopy:(void*)a
                          length:size * sizeof(float)
                         options:MTLResourceStorageModeShared
                     deallocator:nil];
    id<MTLBuffer> buffer_b = [device
        newBufferWithBytesNoCopy:(void*)b
                          length:size * sizeof(float)
                         options:MTLResourceStorageModeShared
                     deallocator:nil];
    id<MTLBuffer> buffer_c = [device
        newBufferWithBytesNoCopy:c
                          length:size * sizeof(float)
                         options:MTLResourceStorageModeShared
                     deallocator:nil];

    dispatchElementwiseKernel(impl_->mul_pipeline, impl_->command_queue,
                              buffer_a, buffer_b, buffer_c,
                              static_cast<uint32_t>(size));

    [buffer_c release];
    [buffer_b release];
    [buffer_a release];
  }
}

void MetalKernels::sub(const float* a, const float* b, float* c, size_t size) {
  @autoreleasepool {
    id<MTLDevice> device = impl_->device;

    id<MTLBuffer> buffer_a = [device
        newBufferWithBytesNoCopy:(void*)a
                          length:size * sizeof(float)
                         options:MTLResourceStorageModeShared
                     deallocator:nil];
    id<MTLBuffer> buffer_b = [device
        newBufferWithBytesNoCopy:(void*)b
                          length:size * sizeof(float)
                         options:MTLResourceStorageModeShared
                     deallocator:nil];
    id<MTLBuffer> buffer_c = [device
        newBufferWithBytesNoCopy:c
                          length:size * sizeof(float)
                         options:MTLResourceStorageModeShared
                     deallocator:nil];

    dispatchElementwiseKernel(impl_->sub_pipeline, impl_->command_queue,
                              buffer_a, buffer_b, buffer_c,
                              static_cast<uint32_t>(size));

    [buffer_c release];
    [buffer_b release];
    [buffer_a release];
  }
}

void MetalKernels::div(const float* a, const float* b, float* c, size_t size) {
  @autoreleasepool {
    id<MTLDevice> device = impl_->device;

    id<MTLBuffer> buffer_a = [device
        newBufferWithBytesNoCopy:(void*)a
                          length:size * sizeof(float)
                         options:MTLResourceStorageModeShared
                     deallocator:nil];
    id<MTLBuffer> buffer_b = [device
        newBufferWithBytesNoCopy:(void*)b
                          length:size * sizeof(float)
                         options:MTLResourceStorageModeShared
                     deallocator:nil];
    id<MTLBuffer> buffer_c = [device
        newBufferWithBytesNoCopy:c
                          length:size * sizeof(float)
                         options:MTLResourceStorageModeShared
                     deallocator:nil];

    dispatchElementwiseKernel(impl_->div_pipeline, impl_->command_queue,
                              buffer_a, buffer_b, buffer_c,
                              static_cast<uint32_t>(size));

    [buffer_c release];
    [buffer_b release];
    [buffer_a release];
  }
}

void MetalKernels::sum(const float* input, float* output, size_t size) {
  @autoreleasepool {
    id<MTLDevice> device = impl_->device;
    id<MTLCommandQueue> queue = impl_->command_queue;

    // Stage 1: Partial sums
    NSUInteger threadGroupSize = 256;
    NSUInteger numGroups = (size + threadGroupSize - 1) / threadGroupSize;

    id<MTLBuffer> input_buffer = [device
        newBufferWithBytesNoCopy:(void*)input
                          length:size * sizeof(float)
                         options:MTLResourceStorageModeShared
                     deallocator:nil];
    id<MTLBuffer> partial_buffer =
        [device newBufferWithLength:numGroups * sizeof(float)
                            options:MTLResourceStorageModeShared];

    // Stage 1 command
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:impl_->sum_stage1_pipeline];
    [encoder setBuffer:input_buffer offset:0 atIndex:0];
    [encoder setBuffer:partial_buffer offset:0 atIndex:1];
    uint32_t size_uint = static_cast<uint32_t>(size);
    [encoder setBytes:&size_uint length:sizeof(uint32_t) atIndex:2];
    [encoder setThreadgroupMemoryLength:threadGroupSize * sizeof(float)
                                atIndex:0];

    MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
    MTLSize groups = MTLSizeMake(numGroups, 1, 1);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Stage 2: Final sum
    id<MTLBuffer> output_buffer = [device
        newBufferWithBytesNoCopy:output
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
    [encoder setThreadgroupMemoryLength:threadGroupSize * sizeof(float)
                                atIndex:0];

    MTLSize finalGroup = MTLSizeMake(threadGroupSize, 1, 1);
    MTLSize oneGroup = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreadgroups:oneGroup threadsPerThreadgroup:finalGroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    [output_buffer release];
    [partial_buffer release];
    [input_buffer release];
  }
}

void MetalKernels::mean(const float* input, float* output, size_t size) {
  // First compute sum
  float* temp_sum = static_cast<float*>(
      impl_->device.newBufferWithLength:sizeof(float)
                                options:MTLResourceStorageModeShared]
          .contents);
  sum(input, temp_sum, size);

  // Then divide by size
  @autoreleasepool {
    id<MTLDevice> device = impl_->device;
    id<MTLCommandQueue> queue = impl_->command_queue;

    id<MTLBuffer> sum_buffer = [device
        newBufferWithBytesNoCopy:temp_sum
                          length:sizeof(float)
                         options:MTLResourceStorageModeShared
                     deallocator:nil];
    id<MTLBuffer> output_buffer = [device
        newBufferWithBytesNoCopy:output
                          length:sizeof(float)
                         options:MTLResourceStorageModeShared
                     deallocator:nil];

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];

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

    [output_buffer release];
    [sum_buffer release];
  }
}

void MetalKernels::matmul(const float* a, const float* b, float* c, size_t m,
                          size_t k, size_t n) {
  @autoreleasepool {
    id<MTLDevice> device = impl_->device;
    id<MTLCommandQueue> queue = impl_->command_queue;

    // Wrap buffers
    id<MTLBuffer> buffer_a = [device
        newBufferWithBytesNoCopy:(void*)a
                          length:m * k * sizeof(float)
                         options:MTLResourceStorageModeShared
                     deallocator:nil];
    id<MTLBuffer> buffer_b = [device
        newBufferWithBytesNoCopy:(void*)b
                          length:k * n * sizeof(float)
                         options:MTLResourceStorageModeShared
                     deallocator:nil];
    id<MTLBuffer> buffer_c = [device
        newBufferWithBytesNoCopy:c
                          length:m * n * sizeof(float)
                         options:MTLResourceStorageModeShared
                     deallocator:nil];

    // Create MPS matrix descriptors
    MPSMatrixDescriptor* desc_a = [MPSMatrixDescriptor
        matrixDescriptorWithRows:m
                         columns:k
                        rowBytes:k * sizeof(float)
                        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* desc_b = [MPSMatrixDescriptor
        matrixDescriptorWithRows:k
                         columns:n
                        rowBytes:n * sizeof(float)
                        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* desc_c = [MPSMatrixDescriptor
        matrixDescriptorWithRows:m
                         columns:n
                        rowBytes:n * sizeof(float)
                        dataType:MPSDataTypeFloat32];

    // Create MPS matrices
    MPSMatrix* matrix_a = [[MPSMatrix alloc] initWithBuffer:buffer_a
                                                 descriptor:desc_a];
    MPSMatrix* matrix_b = [[MPSMatrix alloc] initWithBuffer:buffer_b
                                                 descriptor:desc_b];
    MPSMatrix* matrix_c = [[MPSMatrix alloc] initWithBuffer:buffer_c
                                                 descriptor:desc_c];

    // Create MPS matrix multiplication
    MPSMatrixMultiplication* matmul =
        [[MPSMatrixMultiplication alloc] initWithDevice:device
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

    // Cleanup
    [matmul release];
    [matrix_c release];
    [matrix_b release];
    [matrix_a release];
    [buffer_c release];
    [buffer_b release];
    [buffer_a release];
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
```

---

## 5. CMake 統合

### src/autograd/metal/CMakeLists.txt

```cmake
# Metal カーネルのコンパイル
if(APPLE AND GRADFLOW_ENABLE_METAL)
    message(STATUS "Compiling Metal shaders")

    # Metal ソースファイル
    set(METAL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels.metal
    )

    # Metal Library の出力先
    set(METAL_LIBRARY_OUTPUT "${CMAKE_BINARY_DIR}/kernels.metallib")

    # Metal コンパイルコマンド
    # 1. .metal → .air (Metal Intermediate Representation)
    # 2. .air → .metallib (Metal Library)
    add_custom_command(
        OUTPUT ${METAL_LIBRARY_OUTPUT}
        COMMAND xcrun -sdk macosx metal -c ${METAL_SOURCES} -o ${CMAKE_BINARY_DIR}/kernels.air
        COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/kernels.air -o ${METAL_LIBRARY_OUTPUT}
        DEPENDS ${METAL_SOURCES}
        COMMENT "Compiling Metal shaders"
        VERBATIM
    )

    # Custom target for Metal library
    add_custom_target(metal_kernels ALL DEPENDS ${METAL_LIBRARY_OUTPUT})

    # Install Metal library
    install(FILES ${METAL_LIBRARY_OUTPUT}
            DESTINATION lib
            COMPONENT runtime)

    # gradflow_impl ライブラリに Metal ソースを追加
    target_sources(gradflow_impl PRIVATE
        metal/kernels.mm
    )

    # Metal frameworks をリンク
    target_link_libraries(gradflow_impl PUBLIC
        "-framework Metal"
        "-framework MetalPerformanceShaders"
    )

    # Metal library の依存関係
    add_dependencies(gradflow_impl metal_kernels)
endif()
```

---

## 6. テスト設計

### tests/test_metal_kernels.cpp

```cpp
// tests/test_metal_kernels.cpp
#include <gtest/gtest.h>

#ifdef __APPLE__
#include "gradflow/autograd/metal/device.hpp"
#include "gradflow/autograd/metal/allocator.hpp"
#include "gradflow/autograd/metal/kernels.hpp"
#include <cmath>
#include <numeric>
#include <vector>

using namespace gradflow;
using namespace gradflow::gpu;

class MetalKernelsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!MetalDevice::isAvailable()) {
      GTEST_SKIP() << "Metal is not available on this system";
    }

    device_ = MetalDevice::create();
    ASSERT_NE(device_, nullptr) << "Failed to create Metal device";

    allocator_ = std::make_unique<MetalAllocator>(device_.get());
    kernels_ = std::make_unique<MetalKernels>(device_.get());
  }

  std::unique_ptr<MetalDevice> device_;
  std::unique_ptr<MetalAllocator> allocator_;
  std::unique_ptr<MetalKernels> kernels_;
};

// Test 1: Add カーネル
TEST_F(MetalKernelsTest, AddKernel) {
  constexpr size_t size = 1024;

  // Allocate GPU memory
  float* a = static_cast<float*>(allocator_->allocate(size * sizeof(float)));
  float* b = static_cast<float*>(allocator_->allocate(size * sizeof(float)));
  float* c = static_cast<float*>(allocator_->allocate(size * sizeof(float)));

  // Initialize data (Unified Memory なので CPU から直接書き込み可能)
  for (size_t i = 0; i < size; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i * 2);
  }

  // Execute kernel
  kernels_->add(a, b, c, size);
  kernels_->synchronize();

  // Verify results
  for (size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] + b[i]);
  }

  allocator_->deallocate(c);
  allocator_->deallocate(b);
  allocator_->deallocate(a);
}

// Test 2: Mul カーネル
TEST_F(MetalKernelsTest, MulKernel) {
  constexpr size_t size = 512;

  float* a = static_cast<float*>(allocator_->allocate(size * sizeof(float)));
  float* b = static_cast<float*>(allocator_->allocate(size * sizeof(float)));
  float* c = static_cast<float*>(allocator_->allocate(size * sizeof(float)));

  for (size_t i = 0; i < size; ++i) {
    a[i] = static_cast<float>(i) + 1.0f;
    b[i] = 2.0f;
  }

  kernels_->mul(a, b, c, size);
  kernels_->synchronize();

  for (size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] * b[i]);
  }

  allocator_->deallocate(c);
  allocator_->deallocate(b);
  allocator_->deallocate(a);
}

// Test 3: Sub カーネル
TEST_F(MetalKernelsTest, SubKernel) {
  constexpr size_t size = 256;

  float* a = static_cast<float*>(allocator_->allocate(size * sizeof(float)));
  float* b = static_cast<float*>(allocator_->allocate(size * sizeof(float)));
  float* c = static_cast<float*>(allocator_->allocate(size * sizeof(float)));

  for (size_t i = 0; i < size; ++i) {
    a[i] = static_cast<float>(i * 3);
    b[i] = static_cast<float>(i);
  }

  kernels_->sub(a, b, c, size);
  kernels_->synchronize();

  for (size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] - b[i]);
  }

  allocator_->deallocate(c);
  allocator_->deallocate(b);
  allocator_->deallocate(a);
}

// Test 4: Div カーネル
TEST_F(MetalKernelsTest, DivKernel) {
  constexpr size_t size = 128;

  float* a = static_cast<float*>(allocator_->allocate(size * sizeof(float)));
  float* b = static_cast<float*>(allocator_->allocate(size * sizeof(float)));
  float* c = static_cast<float*>(allocator_->allocate(size * sizeof(float)));

  for (size_t i = 0; i < size; ++i) {
    a[i] = static_cast<float>(i * 4) + 1.0f;
    b[i] = 2.0f;
  }

  kernels_->div(a, b, c, size);
  kernels_->synchronize();

  for (size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] / b[i]);
  }

  allocator_->deallocate(c);
  allocator_->deallocate(b);
  allocator_->deallocate(a);
}

// Test 5: Sum カーネル
TEST_F(MetalKernelsTest, SumKernel) {
  constexpr size_t size = 10000;

  float* input =
      static_cast<float*>(allocator_->allocate(size * sizeof(float)));
  float* output = static_cast<float*>(allocator_->allocate(sizeof(float)));

  // Initialize: input[i] = i
  for (size_t i = 0; i < size; ++i) {
    input[i] = static_cast<float>(i);
  }

  // Expected sum = 0 + 1 + ... + (size-1) = size * (size-1) / 2
  float expected_sum = static_cast<float>(size * (size - 1)) / 2.0f;

  kernels_->sum(input, output, size);
  kernels_->synchronize();

  EXPECT_NEAR(output[0], expected_sum, expected_sum * 1e-5);

  allocator_->deallocate(output);
  allocator_->deallocate(input);
}

// Test 6: Mean カーネル
TEST_F(MetalKernelsTest, MeanKernel) {
  constexpr size_t size = 1000;

  float* input =
      static_cast<float*>(allocator_->allocate(size * sizeof(float)));
  float* output = static_cast<float*>(allocator_->allocate(sizeof(float)));

  // Initialize: input[i] = i * 2
  for (size_t i = 0; i < size; ++i) {
    input[i] = static_cast<float>(i * 2);
  }

  float expected_mean = static_cast<float>(size - 1);  // (0 + 2 + ... + 1998) / 1000

  kernels_->mean(input, output, size);
  kernels_->synchronize();

  EXPECT_NEAR(output[0], expected_mean, expected_mean * 1e-4);

  allocator_->deallocate(output);
  allocator_->deallocate(input);
}

// Test 7: MatMul with MPS (小さい行列)
TEST_F(MetalKernelsTest, MatMulMPS_Small) {
  constexpr size_t m = 4, k = 3, n = 2;

  float* a = static_cast<float*>(allocator_->allocate(m * k * sizeof(float)));
  float* b = static_cast<float*>(allocator_->allocate(k * n * sizeof(float)));
  float* c = static_cast<float*>(allocator_->allocate(m * n * sizeof(float)));

  // A = [[1, 2, 3],
  //      [4, 5, 6],
  //      [7, 8, 9],
  //      [10, 11, 12]]
  for (size_t i = 0; i < m * k; ++i) {
    a[i] = static_cast<float>(i + 1);
  }

  // B = [[1, 2],
  //      [3, 4],
  //      [5, 6]]
  for (size_t i = 0; i < k * n; ++i) {
    b[i] = static_cast<float>(i + 1);
  }

  kernels_->matmul(a, b, c, m, k, n);
  kernels_->synchronize();

  // Expected C = A @ B
  // C[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
  // C[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
  // C[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
  // C[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
  // ... (continue for all elements)

  EXPECT_FLOAT_EQ(c[0 * n + 0], 22.0f);
  EXPECT_FLOAT_EQ(c[0 * n + 1], 28.0f);
  EXPECT_FLOAT_EQ(c[1 * n + 0], 49.0f);
  EXPECT_FLOAT_EQ(c[1 * n + 1], 64.0f);

  allocator_->deallocate(c);
  allocator_->deallocate(b);
  allocator_->deallocate(a);
}

// Test 8: MatMul with MPS (大きい行列 - パフォーマンステスト)
TEST_F(MetalKernelsTest, MatMulMPS_Large) {
  constexpr size_t m = 512, k = 512, n = 512;

  float* a = static_cast<float*>(allocator_->allocate(m * k * sizeof(float)));
  float* b = static_cast<float*>(allocator_->allocate(k * n * sizeof(float)));
  float* c = static_cast<float*>(allocator_->allocate(m * n * sizeof(float)));

  // Initialize with random-like values
  for (size_t i = 0; i < m * k; ++i) {
    a[i] = static_cast<float>(i % 100) / 100.0f;
  }
  for (size_t i = 0; i < k * n; ++i) {
    b[i] = static_cast<float>(i % 100) / 100.0f;
  }

  // Execute
  auto start = std::chrono::high_resolution_clock::now();
  kernels_->matmul(a, b, c, m, k, n);
  kernels_->synchronize();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "MatMul (" << m << "x" << k << " @ " << k << "x" << n
            << ") took " << elapsed.count() << " ms" << std::endl;

  // Verify at least one element (spot check)
  EXPECT_GE(c[0], 0.0f);
  EXPECT_LT(c[0], 1000.0f);

  allocator_->deallocate(c);
  allocator_->deallocate(b);
  allocator_->deallocate(a);
}

// Test 9: CPU と GPU の結果一致検証 (Add)
TEST_F(MetalKernelsTest, CPUGPUConsistency_Add) {
  constexpr size_t size = 2048;

  // CPU 側の計算
  std::vector<float> a_cpu(size), b_cpu(size), c_cpu(size);
  for (size_t i = 0; i < size; ++i) {
    a_cpu[i] = static_cast<float>(i) * 0.5f;
    b_cpu[i] = static_cast<float>(i) * 0.3f;
    c_cpu[i] = a_cpu[i] + b_cpu[i];
  }

  // GPU 側の計算
  float* a_gpu =
      static_cast<float*>(allocator_->allocate(size * sizeof(float)));
  float* b_gpu =
      static_cast<float*>(allocator_->allocate(size * sizeof(float)));
  float* c_gpu =
      static_cast<float*>(allocator_->allocate(size * sizeof(float)));

  std::memcpy(a_gpu, a_cpu.data(), size * sizeof(float));
  std::memcpy(b_gpu, b_cpu.data(), size * sizeof(float));

  kernels_->add(a_gpu, b_gpu, c_gpu, size);
  kernels_->synchronize();

  // 結果比較
  for (size_t i = 0; i < size; ++i) {
    EXPECT_NEAR(c_gpu[i], c_cpu[i], 1e-5);
  }

  allocator_->deallocate(c_gpu);
  allocator_->deallocate(b_gpu);
  allocator_->deallocate(a_gpu);
}

// Test 10: パフォーマンス比較 (GPU vs CPU - large array)
TEST_F(MetalKernelsTest, PerformanceComparison_Add) {
  constexpr size_t size = 10000000;  // 10M elements

  // GPU
  float* a_gpu =
      static_cast<float*>(allocator_->allocate(size * sizeof(float)));
  float* b_gpu =
      static_cast<float*>(allocator_->allocate(size * sizeof(float)));
  float* c_gpu =
      static_cast<float*>(allocator_->allocate(size * sizeof(float)));

  for (size_t i = 0; i < size; ++i) {
    a_gpu[i] = static_cast<float>(i);
    b_gpu[i] = static_cast<float>(i) * 2.0f;
  }

  auto start_gpu = std::chrono::high_resolution_clock::now();
  kernels_->add(a_gpu, b_gpu, c_gpu, size);
  kernels_->synchronize();
  auto end_gpu = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;

  // CPU
  std::vector<float> a_cpu(size), b_cpu(size), c_cpu(size);
  for (size_t i = 0; i < size; ++i) {
    a_cpu[i] = static_cast<float>(i);
    b_cpu[i] = static_cast<float>(i) * 2.0f;
  }

  auto start_cpu = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < size; ++i) {
    c_cpu[i] = a_cpu[i] + b_cpu[i];
  }
  auto end_cpu = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

  std::cout << "Add (" << size << " elements):" << std::endl;
  std::cout << "  GPU: " << gpu_time.count() << " ms" << std::endl;
  std::cout << "  CPU: " << cpu_time.count() << " ms" << std::endl;
  std::cout << "  Speedup: " << cpu_time.count() / gpu_time.count() << "x"
            << std::endl;

  // GPU should be faster for large arrays
  EXPECT_LT(gpu_time.count(), cpu_time.count());

  allocator_->deallocate(c_gpu);
  allocator_->deallocate(b_gpu);
  allocator_->deallocate(a_gpu);
}

#endif  // __APPLE__
```

---

## 7. 実装タスクリスト（github-issue-implementer への指示）

### Phase 1: Metal Shading Language カーネルの実装

1. **`src/autograd/metal/kernels.metal` を作成**
   - Elementwise カーネル: `add_kernel`, `mul_kernel`, `sub_kernel`, `div_kernel`
   - Reduction カーネル: `sum_kernel_stage1`, `sum_kernel_stage2`, `mean_kernel`
   - Thread Group Size: 256 を使用
   - Bounds checking を含める (`if (gid < size)`)

### Phase 2: C++ Public Interface の作成

2. **`include/gradflow/autograd/metal/kernels.hpp` を作成**
   - `MetalKernels` クラスの宣言
   - すべての public メソッドのドキュメント
   - Forward declaration (`MetalKernelsImpl`)

### Phase 3: C++ ラッパーの実装

3. **`src/autograd/metal/kernels.mm` を作成**
   - `MetalKernelsImpl` クラス (Objective-C オブジェクトを保持)
   - Metal Library のロード
   - Compute Pipeline State の作成
   - 各カーネルの実行メソッド (`add`, `mul`, `sub`, `div`, `sum`, `mean`)
   - MPS による `matmul` 実装

### Phase 4: CMake 統合

4. **`src/autograd/metal/CMakeLists.txt` を更新**
   - Metal Shading Language のコンパイルコマンド
   - `xcrun metal` による .air → .metallib の生成
   - MetalPerformanceShaders framework のリンク

### Phase 5: テストの実装

5. **`tests/test_metal_kernels.cpp` を作成**
   - MetalKernelsTest::AddKernel
   - MetalKernelsTest::MulKernel
   - MetalKernelsTest::SubKernel
   - MetalKernelsTest::DivKernel
   - MetalKernelsTest::SumKernel
   - MetalKernelsTest::MeanKernel
   - MetalKernelsTest::MatMulMPS_Small
   - MetalKernelsTest::MatMulMPS_Large
   - MetalKernelsTest::CPUGPUConsistency_Add
   - MetalKernelsTest::PerformanceComparison_Add

### Phase 6: ドキュメント

6. **README.md を更新**
   - Metal Compute Shader の使用方法
   - パフォーマンスベンチマーク結果

---

## 8. 完了基準

- [ ] すべてのテストが pass (10 個のテスト)
- [ ] CPU と GPU の計算結果が一致（数値誤差 < 1e-5）
- [ ] GPU が CPU より高速（10M 要素の add で speedup > 1.0x）
- [ ] MatMul が MPS の 90% 以上の速度（実際には MPS を直接使用するため 100%）
- [ ] Metal Library が正常にロードされる
- [ ] CI で Metal テストが実行される (macOS runner)

---

## 9. トレードオフと設計判断

### メリット

1. **高パフォーマンス**: Metal Performance Shaders により、最適化された行列乗算を実現
2. **Unified Memory の活用**: CPU/GPU 間のデータ転送オーバーヘッドを最小化
3. **拡張性**: 新しいカーネルを追加する際、.metal ファイルと対応する C++ ラッパーを追加するだけ
4. **テスタビリティ**: CPU と GPU の結果を比較することで、数値的正確性を検証
5. **Apple Silicon 最適化**: Thread Group Size やメモリアクセスパターンを最適化

### リスク・注意点

1. **プラットフォーム依存**: macOS/iOS 専用で、他のプラットフォームでは動作しない
2. **Metal Library のパス**: Runtime 時に .metallib ファイルを正しくロードする必要がある
3. **MPS の C++ 統合**: MPSMatrixMultiplication は Objective-C API のため、Objective-C++ ブリッジが必要
4. **パフォーマンスプロファイリング**: 実際の speedup は測定して検証する必要がある
5. **小さい配列**: 非常に小さい配列では、GPU 起動のオーバーヘッドにより CPU の方が高速な可能性がある

---

## 10. 参考資料とリンク

### Apple 公式ドキュメント

- [Learn performance best practices for Metal shaders - Tech Talks](https://developer.apple.com/videos/play/tech-talks/111373/)
- [Optimize Metal Performance for Apple silicon Macs - WWDC20](https://developer.apple.com/videos/play/wwdc2020/10632/)
- [Calculating threadgroup and grid sizes | Apple Developer Documentation](https://developer.apple.com/documentation/metal/compute_passes/calculating_threadgroup_and_grid_sizes)
- [Metal Shading Language Specification Version 4](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [MPSMatrixMultiplication | Apple Developer Documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)

### 実装例とチュートリアル

- [GitHub - bkvogel/metal_performance_testing](https://github.com/bkvogel/metal_performance_testing)
- [Matrix Multiplication with Metal Performance Shaders](https://machinethink.net/blog/mps-matrix-multiplication/)
- [Understanding Metal Shading Language (MSL) Fundamentals](https://medium.com/@er.pwndhull07/understanding-metal-shading-language-msl-fundamentals2-8-b6bd99a48bb3)

---

以上が Issue #15 の詳細設計書です。この設計に基づき、github-issue-implementer が実装を進めることで、GradFlow に Metal Compute Shader サポートが追加されます。
