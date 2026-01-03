# Issue #19: CPU 最適化の詳細設計書

## 1. 調査・リサーチ結果

### CPU 最適化のベストプラクティス

#### 1.1 SIMD ベクトル化（AVX2/AVX512）

**SIMD (Single Instruction Multiple Data)** は、単一の命令で複数のデータ要素を同時に処理する技術です。現代の x86_64 プロセッサでは、以下の SIMD 命令セットが利用可能です：

- **SSE (Streaming SIMD Extensions)**: 128-bit レジスタ、4 個の float を同時処理
- **AVX (Advanced Vector Extensions)**: 256-bit レジスタ、8 個の float を同時処理
- **AVX2**: AVX の拡張、整数演算のサポート強化
- **AVX-512**: 512-bit レジスタ、16 個の float を同時処理（Intel Xeon, Ice Lake 以降）

**最適化原則**:
1. **アライメント**: メモリアドレスを SIMD レジスタサイズ（16/32/64 バイト）に整列させることで、ロード/ストア命令の効率を向上
2. **ベクトル幅の選択**: AVX2 は広くサポートされており、互換性とパフォーマンスのバランスが良い
3. **コンパイラ最適化**: `-O3 -march=native -mavx2` などのフラグで自動ベクトル化を促進
4. **Intrinsic 関数**: `<immintrin.h>` の intrinsic 関数を使用して明示的な SIMD コードを記述

#### 1.2 OpenMP による並列化

**OpenMP** は、共有メモリマルチプロセッサシステム向けの並列プログラミング API です。

**ベストプラクティス**:
1. **並列領域の粒度**: ループのオーバーヘッドを考慮し、十分な作業量を持つループを並列化
2. **スケジューリング**: `schedule(static)` はキャッシュ効率が良く、`schedule(dynamic)` は負荷分散に有効
3. **False Sharing の回避**: 異なるスレッドが同一キャッシュライン上の異なるデータを更新することで生じる性能低下を防ぐ
4. **スレッド数の調整**: `omp_set_num_threads()` または環境変数 `OMP_NUM_THREADS` で最適なスレッド数を設定

#### 1.3 Blocked MatMul（キャッシュ効率の改善）

**Blocked Matrix Multiplication** は、行列を小さなブロックに分割し、ブロック単位で計算することでキャッシュヒット率を向上させる手法です。

**キャッシュ階層の理解**:
- **L1 キャッシュ**: 32-64 KB、最も高速（1-2 サイクル）
- **L2 キャッシュ**: 256-512 KB、高速（10-20 サイクル）
- **L3 キャッシュ**: 数 MB～数十 MB、中速（40-75 サイクル）
- **メインメモリ**: 数 GB、低速（100-300 サイクル）

**最適化戦略**:
1. **ブロックサイズの選択**: L1 キャッシュに 3 つのブロック（A, B, C）が収まるサイズを選択（例: 32x32, 64x64）
2. **レジスタブロッキング**: さらに小さなマイクロカーネル（例: 4x4）を使用してレジスタを効率的に活用
3. **ループの順序**: メモリアクセスパターンを最適化（例: ikj ループの順序）
4. **Prefetching**: `_mm_prefetch()` で次のデータをキャッシュに事前ロード

#### 1.4 メモリアライメントの最適化

**メモリアライメント** は、データ構造をメモリアドレスの境界に整列させることで、アクセス効率を向上させます。

**アライメント要件**:
- **SSE**: 16 バイトアライメント
- **AVX/AVX2**: 32 バイトアライメント
- **AVX-512**: 64 バイトアライメント

**実装方法**:
1. **動的アライメント**: `aligned_alloc()`, `posix_memalign()`, または `_mm_malloc()`
2. **静的アライメント**: `alignas(32)` 指定子
3. **コンパイラヒント**: `__assume_aligned()` または `__builtin_assume_aligned()`

### 参考文献

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) - Intel SIMD intrinsic 関数の公式リファレンス
- [OpenMP 5.2 Specification](https://www.openmp.org/specifications/) - OpenMP の公式仕様
- [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf) - メモリ階層とキャッシュの詳細解説
- [Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf) - 高性能行列乗算の実装手法
- [Optimizing Matrix Multiply (UC Berkeley CS267)](https://people.eecs.berkeley.edu/~demmel/cs267_Spr16/Lectures/lecture11_matmul_jwd16_4pp.pdf) - 行列乗算の最適化講義資料

---

## 2. 分析と評価

### 現状の課題

- 現在の CPU 実装は単純なループベースで、SIMD やマルチスレッド最適化が未実装
- 既存の `matmul()` 関数（`include/gradflow/autograd/ops/matmul.hpp`）は naive な O(n³) アルゴリズム
- Metal GPU 実装は Phase 3 で完了しているが、CPU パフォーマンスとの差が大きい
- 大規模なテンソル演算（例: 1024x1024 行列乗算）では、CPU が GPU に大きく劣る

### 設計原則

#### 1. 単一責任の原則 (SRP)

- **CPUKernels**: CPU 最適化カーネルの管理のみを担当
- **SIMDOps**: SIMD 命令を使用した低レベル演算を提供
- **BlockedMatMul**: ブロックベース行列乗算の実装
- 各演算（add, mul, matmul）は独立した関数として実装

#### 2. 開放閉鎖原則 (OCP)

- 新しい最適化手法（例: AVX-512, ARM NEON）を追加する際、既存コードを変更せずに拡張可能
- コンパイル時の機能検出により、利用可能な SIMD 命令セットを自動選択

#### 3. 依存性逆転の原則 (DIP)

- 高レベルモジュール（Tensor, Operation）は CPU 固有の詳細に依存せず、抽象インターフェースを通じて利用
- SIMD の詳細は実装ファイル内に隠蔽

#### 4. パフォーマンス最優先

- **ベンチマーク駆動**: すべての最適化は測定可能なパフォーマンス向上を伴う
- **プロファイリング**: `perf`, `Instruments`, または `VTune` を使用してボトルネックを特定
- **段階的最適化**: 各最適化手法の効果を個別に測定

---

## 3. 推奨アーキテクチャ案

### 設計のコンセプト

**3 層アーキテクチャ**:

1. **SIMD Intrinsic Layer** (`src/autograd/cpu/simd_ops.hpp`): SIMD intrinsic 関数によるベクトル演算
2. **Optimized Kernels Layer** (`src/autograd/cpu/kernels_avx2.cpp`): 最適化された演算カーネル
3. **Public Interface Layer** (`include/gradflow/autograd/cpu/kernels.hpp`): クリーンな C++ API

**主要コンポーネント**:

- **CPUKernels**: すべての最適化 CPU カーネルを管理するクラス
- **SIMDOps**: AVX2 intrinsic を使用した低レベル演算
- **BlockedMatMul**: キャッシュ効率を考慮した行列乗算
- **AlignedAllocator**: アライメント付きメモリ確保

### クラス設計

#### 3.1 CPUKernels クラス（C++ Public Interface）

```cpp
// include/gradflow/autograd/cpu/kernels.hpp
#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace gradflow {
namespace cpu {

/**
 * @brief CPU 最適化カーネルの管理クラス
 *
 * CPU 上での高速な演算カーネルを提供します。
 * - SIMD ベクトル化（AVX2/AVX512）
 * - OpenMP による並列化
 * - Blocked MatMul（キャッシュ効率の改善）
 * - メモリアライメントの最適化
 *
 * 使用例:
 * @code
 *   CPUKernels kernels;
 *
 *   float* a = alignedAlloc(1024 * sizeof(float));
 *   float* b = alignedAlloc(1024 * sizeof(float));
 *   float* c = alignedAlloc(1024 * sizeof(float));
 *
 *   kernels.add(a, b, c, 1024);  // c = a + b (SIMD 最適化)
 * @endcode
 */
class CPUKernels {
 public:
  /**
   * @brief CPUKernels を構築
   *
   * 利用可能な SIMD 命令セットを自動検出します。
   */
  CPUKernels();

  ~CPUKernels();

  // コピー・ムーブ禁止
  CPUKernels(const CPUKernels&) = delete;
  CPUKernels& operator=(const CPUKernels&) = delete;
  CPUKernels(CPUKernels&&) = delete;
  CPUKernels& operator=(CPUKernels&&) = delete;

  // ===== Elementwise Operations (SIMD Optimized) =====

  /**
   * @brief 要素ごとの加算: c = a + b (SIMD 最適化)
   *
   * @param a 入力配列 A (32-byte aligned)
   * @param b 入力配列 B (32-byte aligned)
   * @param c 出力配列 C (32-byte aligned)
   * @param size 要素数
   */
  void add(const float* a, const float* b, float* c, size_t size);

  /**
   * @brief 要素ごとの乗算: c = a * b (SIMD 最適化)
   *
   * @param a 入力配列 A (32-byte aligned)
   * @param b 入力配列 B (32-byte aligned)
   * @param c 出力配列 C (32-byte aligned)
   * @param size 要素数
   */
  void mul(const float* a, const float* b, float* c, size_t size);

  /**
   * @brief 要素ごとの減算: c = a - b (SIMD 最適化)
   *
   * @param a 入力配列 A (32-byte aligned)
   * @param b 入力配列 B (32-byte aligned)
   * @param c 出力配列 C (32-byte aligned)
   * @param size 要素数
   */
  void sub(const float* a, const float* b, float* c, size_t size);

  /**
   * @brief 要素ごとの除算: c = a / b (SIMD 最適化)
   *
   * @param a 入力配列 A (32-byte aligned)
   * @param b 入力配列 B (32-byte aligned)
   * @param c 出力配列 C (32-byte aligned)
   * @param size 要素数
   */
  void div(const float* a, const float* b, float* c, size_t size);

  // ===== Matrix Operations (Blocked MatMul) =====

  /**
   * @brief 行列乗算: C = A * B (Blocked MatMul + OpenMP 並列化)
   *
   * ブロックベースのアルゴリズムでキャッシュ効率を改善し、
   * OpenMP で並列化します。
   *
   * @param a 入力行列 A (32-byte aligned, row-major)
   * @param b 入力行列 B (32-byte aligned, row-major)
   * @param c 出力行列 C (32-byte aligned, row-major)
   * @param m 行列 A の行数
   * @param k 行列 A の列数 / 行列 B の行数
   * @param n 行列 B の列数
   */
  void matmul(const float* a, const float* b, float* c, size_t m, size_t k,
              size_t n);

  /**
   * @brief 利用可能な SIMD 命令セットを取得
   *
   * @return SIMD 命令セット名（例: "AVX2", "AVX512", "SSE"）
   */
  std::string getSIMDInfo() const;

 private:
  bool has_avx2_;
  bool has_avx512_;
};

/**
 * @brief アライメント付きメモリ確保
 *
 * @param size 確保するバイト数
 * @param alignment アライメント（デフォルト: 32 バイト）
 * @return アライメントされたメモリポインタ
 */
void* alignedAlloc(size_t size, size_t alignment = 32);

/**
 * @brief アライメント付きメモリ解放
 *
 * @param ptr alignedAlloc で確保したポインタ
 */
void alignedFree(void* ptr);

}  // namespace cpu
}  // namespace gradflow
```

---

## 4. 実装の詳細

### 4.1 SIMD Intrinsic 演算 (simd_ops.hpp)

```cpp
// src/autograd/cpu/simd_ops.hpp
#pragma once

#include <immintrin.h>  // AVX2/AVX512 intrinsics
#include <cstddef>

namespace gradflow {
namespace cpu {
namespace simd {

/**
 * @brief AVX2 を使用した要素ごとの加算
 *
 * 32 バイトアライメント必須。サイズは 8 の倍数であることを推奨。
 */
inline void add_avx2(const float* a, const float* b, float* c, size_t size) {
  size_t i = 0;

  // AVX2: 8 個の float を同時処理
  for (; i + 8 <= size; i += 8) {
    __m256 va = _mm256_load_ps(a + i);  // Aligned load
    __m256 vb = _mm256_load_ps(b + i);
    __m256 vc = _mm256_add_ps(va, vb);
    _mm256_store_ps(c + i, vc);         // Aligned store
  }

  // 残りの要素をスカラー処理
  for (; i < size; ++i) {
    c[i] = a[i] + b[i];
  }
}

/**
 * @brief AVX2 を使用した要素ごとの乗算
 */
inline void mul_avx2(const float* a, const float* b, float* c, size_t size) {
  size_t i = 0;

  for (; i + 8 <= size; i += 8) {
    __m256 va = _mm256_load_ps(a + i);
    __m256 vb = _mm256_load_ps(b + i);
    __m256 vc = _mm256_mul_ps(va, vb);
    _mm256_store_ps(c + i, vc);
  }

  for (; i < size; ++i) {
    c[i] = a[i] * b[i];
  }
}

/**
 * @brief AVX2 を使用した要素ごとの減算
 */
inline void sub_avx2(const float* a, const float* b, float* c, size_t size) {
  size_t i = 0;

  for (; i + 8 <= size; i += 8) {
    __m256 va = _mm256_load_ps(a + i);
    __m256 vb = _mm256_load_ps(b + i);
    __m256 vc = _mm256_sub_ps(va, vb);
    _mm256_store_ps(c + i, vc);
  }

  for (; i < size; ++i) {
    c[i] = a[i] - b[i];
  }
}

/**
 * @brief AVX2 を使用した要素ごとの除算
 */
inline void div_avx2(const float* a, const float* b, float* c, size_t size) {
  size_t i = 0;

  for (; i + 8 <= size; i += 8) {
    __m256 va = _mm256_load_ps(a + i);
    __m256 vb = _mm256_load_ps(b + i);
    __m256 vc = _mm256_div_ps(va, vb);
    _mm256_store_ps(c + i, vc);
  }

  for (; i < size; ++i) {
    c[i] = a[i] / b[i];
  }
}

}  // namespace simd
}  // namespace cpu
}  // namespace gradflow
```

### 4.2 Blocked MatMul (matmul_blocked.cpp)

```cpp
// src/autograd/cpu/matmul_blocked.cpp
#include "gradflow/autograd/cpu/kernels.hpp"
#include <omp.h>
#include <algorithm>
#include <cstring>

namespace gradflow {
namespace cpu {

namespace {

// ブロックサイズ（L1 キャッシュに収まるサイズ）
constexpr size_t kBlockSize = 64;

// マイクロカーネル: 4x4 レジスタブロック
inline void matmul_kernel_4x4(const float* a, const float* b, float* c,
                               size_t k, size_t lda, size_t ldb, size_t ldc) {
  // 4x4 のレジスタブロックを使用した高速行列乗算
  float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
  float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
  float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
  float c30 = 0, c31 = 0, c32 = 0, c33 = 0;

  for (size_t p = 0; p < k; ++p) {
    float a0 = a[0 * lda + p];
    float a1 = a[1 * lda + p];
    float a2 = a[2 * lda + p];
    float a3 = a[3 * lda + p];

    float b0 = b[p * ldb + 0];
    float b1 = b[p * ldb + 1];
    float b2 = b[p * ldb + 2];
    float b3 = b[p * ldb + 3];

    c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
    c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
    c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
    c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
  }

  c[0 * ldc + 0] += c00; c[0 * ldc + 1] += c01; c[0 * ldc + 2] += c02; c[0 * ldc + 3] += c03;
  c[1 * ldc + 0] += c10; c[1 * ldc + 1] += c11; c[1 * ldc + 2] += c12; c[1 * ldc + 3] += c13;
  c[2 * ldc + 0] += c20; c[2 * ldc + 1] += c21; c[2 * ldc + 2] += c22; c[2 * ldc + 3] += c23;
  c[3 * ldc + 0] += c30; c[3 * ldc + 1] += c31; c[3 * ldc + 2] += c32; c[3 * ldc + 3] += c33;
}

// ブロック行列乗算: C_block = A_block * B_block
void matmul_block(const float* a, const float* b, float* c,
                  size_t m, size_t k, size_t n,
                  size_t lda, size_t ldb, size_t ldc) {
  // 4x4 マイクロカーネルを使用
  size_t i = 0;
  for (; i + 4 <= m; i += 4) {
    size_t j = 0;
    for (; j + 4 <= n; j += 4) {
      matmul_kernel_4x4(a + i * lda, b + j, c + i * ldc + j,
                        k, lda, ldb, ldc);
    }

    // 残りの列をスカラー処理
    for (; j < n; ++j) {
      for (size_t ii = i; ii < i + 4 && ii < m; ++ii) {
        float sum = 0;
        for (size_t p = 0; p < k; ++p) {
          sum += a[ii * lda + p] * b[p * ldb + j];
        }
        c[ii * ldc + j] += sum;
      }
    }
  }

  // 残りの行をスカラー処理
  for (; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      float sum = 0;
      for (size_t p = 0; p < k; ++p) {
        sum += a[i * lda + p] * b[p * ldb + j];
      }
      c[i * ldc + j] += sum;
    }
  }
}

}  // namespace

void CPUKernels::matmul(const float* a, const float* b, float* c,
                        size_t m, size_t k, size_t n) {
  // C を 0 で初期化
  std::memset(c, 0, m * n * sizeof(float));

  // OpenMP 並列化: ブロックごとに並列実行
  #pragma omp parallel for collapse(2) schedule(static)
  for (size_t i = 0; i < m; i += kBlockSize) {
    for (size_t j = 0; j < n; j += kBlockSize) {
      for (size_t p = 0; p < k; p += kBlockSize) {
        size_t block_m = std::min(kBlockSize, m - i);
        size_t block_k = std::min(kBlockSize, k - p);
        size_t block_n = std::min(kBlockSize, n - j);

        matmul_block(a + i * k + p, b + p * n + j, c + i * n + j,
                     block_m, block_k, block_n, k, n, n);
      }
    }
  }
}

}  // namespace cpu
}  // namespace gradflow
```

### 4.3 CPUKernels 実装 (kernels_avx2.cpp)

```cpp
// src/autograd/cpu/kernels_avx2.cpp
#include "gradflow/autograd/cpu/kernels.hpp"
#include "simd_ops.hpp"
#include <cpuid.h>
#include <cstdlib>
#include <cstring>

namespace gradflow {
namespace cpu {

// CPUID による機能検出
static bool has_avx2() {
#if defined(__x86_64__) || defined(_M_X64)
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid(7, &eax, &ebx, &ecx, &edx)) {
    return (ebx & (1 << 5)) != 0;  // AVX2 bit
  }
#endif
  return false;
}

static bool has_avx512() {
#if defined(__x86_64__) || defined(_M_X64)
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid(7, &eax, &ebx, &ecx, &edx)) {
    return (ebx & (1 << 16)) != 0;  // AVX-512F bit
  }
#endif
  return false;
}

CPUKernels::CPUKernels()
    : has_avx2_(has_avx2()), has_avx512_(has_avx512()) {}

CPUKernels::~CPUKernels() = default;

void CPUKernels::add(const float* a, const float* b, float* c, size_t size) {
  if (has_avx2_) {
    simd::add_avx2(a, b, c, size);
  } else {
    // Fallback: スカラー実装
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] + b[i];
    }
  }
}

void CPUKernels::mul(const float* a, const float* b, float* c, size_t size) {
  if (has_avx2_) {
    simd::mul_avx2(a, b, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] * b[i];
    }
  }
}

void CPUKernels::sub(const float* a, const float* b, float* c, size_t size) {
  if (has_avx2_) {
    simd::sub_avx2(a, b, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] - b[i];
    }
  }
}

void CPUKernels::div(const float* a, const float* b, float* c, size_t size) {
  if (has_avx2_) {
    simd::div_avx2(a, b, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] / b[i];
    }
  }
}

std::string CPUKernels::getSIMDInfo() const {
  if (has_avx512_) {
    return "AVX-512";
  } else if (has_avx2_) {
    return "AVX2";
  } else {
    return "Scalar (No SIMD)";
  }
}

void* alignedAlloc(size_t size, size_t alignment) {
#if defined(_MSC_VER)
  return _aligned_malloc(size, alignment);
#else
  void* ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size) != 0) {
    return nullptr;
  }
  return ptr;
#endif
}

void alignedFree(void* ptr) {
#if defined(_MSC_VER)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

}  // namespace cpu
}  // namespace gradflow
```

---

## 5. CMake 統合

### src/autograd/cpu/CMakeLists.txt

```cmake
# CPU 最適化カーネルの設定
if(GRADFLOW_ENABLE_CPU_OPTIMIZATIONS)
    message(STATUS "Enabling CPU optimizations")

    # AVX2 フラグ
    if(MSVC)
        set(AVX2_FLAGS /arch:AVX2)
    else()
        set(AVX2_FLAGS -mavx2 -mfma)
    endif()

    # OpenMP サポート
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    else()
        message(WARNING "OpenMP not found, parallel MatMul will be disabled")
    endif()

    # CPU 最適化カーネルのソース
    set(CPU_OPT_SOURCES
        cpu/kernels_avx2.cpp
        cpu/matmul_blocked.cpp
    )

    # gradflow_impl ライブラリに CPU 最適化ソースを追加
    target_sources(gradflow_impl PRIVATE ${CPU_OPT_SOURCES})

    # AVX2 コンパイルフラグを追加
    target_compile_options(gradflow_impl PRIVATE ${AVX2_FLAGS})

    # OpenMP をリンク
    if(OpenMP_CXX_FOUND)
        target_link_libraries(gradflow_impl PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()
```

### ルート CMakeLists.txt への追加

```cmake
# Build Options
option(GRADFLOW_ENABLE_CPU_OPTIMIZATIONS "Enable CPU optimizations (SIMD, OpenMP)" ON)
```

---

## 6. テスト設計

### tests/test_cpu_optimized.cpp

```cpp
// tests/test_cpu_optimized.cpp
#include <gtest/gtest.h>
#include "gradflow/autograd/cpu/kernels.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace gradflow::cpu;

class CPUOptimizedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    kernels_ = std::make_unique<CPUKernels>();
    std::cout << "SIMD Support: " << kernels_->getSIMDInfo() << std::endl;
  }

  std::unique_ptr<CPUKernels> kernels_;
};

// Test 1: Add SIMD 最適化の正確性
TEST_F(CPUOptimizedTest, AddSIMD_Correctness) {
  constexpr size_t size = 1024;

  float* a = static_cast<float*>(alignedAlloc(size * sizeof(float)));
  float* b = static_cast<float*>(alignedAlloc(size * sizeof(float)));
  float* c = static_cast<float*>(alignedAlloc(size * sizeof(float)));

  for (size_t i = 0; i < size; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i * 2);
  }

  kernels_->add(a, b, c, size);

  for (size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] + b[i]);
  }

  alignedFree(c);
  alignedFree(b);
  alignedFree(a);
}

// Test 2: Mul SIMD 最適化の正確性
TEST_F(CPUOptimizedTest, MulSIMD_Correctness) {
  constexpr size_t size = 512;

  float* a = static_cast<float*>(alignedAlloc(size * sizeof(float)));
  float* b = static_cast<float*>(alignedAlloc(size * sizeof(float)));
  float* c = static_cast<float*>(alignedAlloc(size * sizeof(float)));

  for (size_t i = 0; i < size; ++i) {
    a[i] = static_cast<float>(i) + 1.0f;
    b[i] = 2.0f;
  }

  kernels_->mul(a, b, c, size);

  for (size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] * b[i]);
  }

  alignedFree(c);
  alignedFree(b);
  alignedFree(a);
}

// Test 3: Blocked MatMul の正確性
TEST_F(CPUOptimizedTest, BlockedMatMul_Correctness) {
  constexpr size_t m = 4, k = 3, n = 2;

  float* a = static_cast<float*>(alignedAlloc(m * k * sizeof(float)));
  float* b = static_cast<float*>(alignedAlloc(k * n * sizeof(float)));
  float* c = static_cast<float*>(alignedAlloc(m * n * sizeof(float)));

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

  // Expected C = A @ B
  EXPECT_FLOAT_EQ(c[0 * n + 0], 22.0f);  // 1*1 + 2*3 + 3*5
  EXPECT_FLOAT_EQ(c[0 * n + 1], 28.0f);  // 1*2 + 2*4 + 3*6
  EXPECT_FLOAT_EQ(c[1 * n + 0], 49.0f);  // 4*1 + 5*3 + 6*5
  EXPECT_FLOAT_EQ(c[1 * n + 1], 64.0f);  // 4*2 + 5*4 + 6*6

  alignedFree(c);
  alignedFree(b);
  alignedFree(a);
}

// Test 4: 最適化前との性能比較 (Add)
TEST_F(CPUOptimizedTest, Performance_Add) {
  constexpr size_t size = 10000000;  // 10M elements

  float* a = static_cast<float*>(alignedAlloc(size * sizeof(float)));
  float* b = static_cast<float*>(alignedAlloc(size * sizeof(float)));
  float* c_opt = static_cast<float*>(alignedAlloc(size * sizeof(float)));
  float* c_naive = static_cast<float*>(alignedAlloc(size * sizeof(float)));

  for (size_t i = 0; i < size; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i) * 2.0f;
  }

  // Optimized
  auto start_opt = std::chrono::high_resolution_clock::now();
  kernels_->add(a, b, c_opt, size);
  auto end_opt = std::chrono::high_resolution_clock::now();

  // Naive
  auto start_naive = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < size; ++i) {
    c_naive[i] = a[i] + b[i];
  }
  auto end_naive = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> opt_time = end_opt - start_opt;
  std::chrono::duration<double, std::milli> naive_time = end_naive - start_naive;

  std::cout << "Add (" << size << " elements):" << std::endl;
  std::cout << "  Optimized: " << opt_time.count() << " ms" << std::endl;
  std::cout << "  Naive: " << naive_time.count() << " ms" << std::endl;
  std::cout << "  Speedup: " << naive_time.count() / opt_time.count() << "x" << std::endl;

  // 最適化版が高速であることを確認
  EXPECT_LT(opt_time.count(), naive_time.count());

  alignedFree(c_naive);
  alignedFree(c_opt);
  alignedFree(b);
  alignedFree(a);
}

// Test 5: MatMul の性能比較
TEST_F(CPUOptimizedTest, Performance_MatMul) {
  constexpr size_t m = 512, k = 512, n = 512;

  float* a = static_cast<float*>(alignedAlloc(m * k * sizeof(float)));
  float* b = static_cast<float*>(alignedAlloc(k * n * sizeof(float)));
  float* c_opt = static_cast<float*>(alignedAlloc(m * n * sizeof(float)));
  float* c_naive = static_cast<float*>(alignedAlloc(m * n * sizeof(float)));

  for (size_t i = 0; i < m * k; ++i) {
    a[i] = static_cast<float>(i % 100) / 100.0f;
  }
  for (size_t i = 0; i < k * n; ++i) {
    b[i] = static_cast<float>(i % 100) / 100.0f;
  }

  // Optimized (Blocked + OpenMP)
  auto start_opt = std::chrono::high_resolution_clock::now();
  kernels_->matmul(a, b, c_opt, m, k, n);
  auto end_opt = std::chrono::high_resolution_clock::now();

  // Naive (triple loop)
  auto start_naive = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      float sum = 0;
      for (size_t p = 0; p < k; ++p) {
        sum += a[i * k + p] * b[p * n + j];
      }
      c_naive[i * n + j] = sum;
    }
  }
  auto end_naive = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> opt_time = end_opt - start_opt;
  std::chrono::duration<double, std::milli> naive_time = end_naive - start_naive;

  std::cout << "MatMul (" << m << "x" << k << " @ " << k << "x" << n << "):" << std::endl;
  std::cout << "  Optimized: " << opt_time.count() << " ms" << std::endl;
  std::cout << "  Naive: " << naive_time.count() << " ms" << std::endl;
  std::cout << "  Speedup: " << naive_time.count() / opt_time.count() << "x" << std::endl;

  // 最適化版が高速であることを確認
  EXPECT_LT(opt_time.count(), naive_time.count());

  // 数値的正確性の確認（スポットチェック）
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_NEAR(c_opt[i], c_naive[i], 1e-3);
  }

  alignedFree(c_naive);
  alignedFree(c_opt);
  alignedFree(b);
  alignedFree(a);
}

// Test 6: Eigen との性能比較
#ifdef EIGEN3_FOUND
#include <Eigen/Dense>

TEST_F(CPUOptimizedTest, Performance_MatMul_vs_Eigen) {
  constexpr size_t m = 512, k = 512, n = 512;

  float* a = static_cast<float*>(alignedAlloc(m * k * sizeof(float)));
  float* b = static_cast<float*>(alignedAlloc(k * n * sizeof(float)));
  float* c_opt = static_cast<float*>(alignedAlloc(m * n * sizeof(float)));

  for (size_t i = 0; i < m * k; ++i) {
    a[i] = static_cast<float>(i % 100) / 100.0f;
  }
  for (size_t i = 0; i < k * n; ++i) {
    b[i] = static_cast<float>(i % 100) / 100.0f;
  }

  // Optimized
  auto start_opt = std::chrono::high_resolution_clock::now();
  kernels_->matmul(a, b, c_opt, m, k, n);
  auto end_opt = std::chrono::high_resolution_clock::now();

  // Eigen
  Eigen::MatrixXf A = Eigen::Map<Eigen::MatrixXf>(a, k, m).transpose();
  Eigen::MatrixXf B = Eigen::Map<Eigen::MatrixXf>(b, n, k).transpose();

  auto start_eigen = std::chrono::high_resolution_clock::now();
  Eigen::MatrixXf C = A * B;
  auto end_eigen = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> opt_time = end_opt - start_opt;
  std::chrono::duration<double, std::milli> eigen_time = end_eigen - start_eigen;

  std::cout << "MatMul vs Eigen (" << m << "x" << k << " @ " << k << "x" << n << "):" << std::endl;
  std::cout << "  Optimized: " << opt_time.count() << " ms" << std::endl;
  std::cout << "  Eigen: " << eigen_time.count() << " ms" << std::endl;
  std::cout << "  Relative Performance: " << (opt_time.count() / eigen_time.count()) * 100.0 << "%" << std::endl;

  // 目標: Eigen の 80% 以上の速度
  EXPECT_LT(opt_time.count(), eigen_time.count() * 1.25);

  alignedFree(c_opt);
  alignedFree(b);
  alignedFree(a);
}
#endif
```

---

## 7. 実装タスクリスト（github-issue-implementer への指示）

### Phase 1: SIMD Intrinsic 演算の実装

1. **`src/autograd/cpu/simd_ops.hpp` を作成**
   - AVX2 intrinsic を使用した elementwise 演算
   - `add_avx2`, `mul_avx2`, `sub_avx2`, `div_avx2`
   - アライメント前提、残余要素のスカラー処理

### Phase 2: Blocked MatMul の実装

2. **`src/autograd/cpu/matmul_blocked.cpp` を作成**
   - ブロックサイズ: 64x64（L1 キャッシュに収まるサイズ）
   - マイクロカーネル: 4x4 レジスタブロック
   - OpenMP 並列化（`#pragma omp parallel for collapse(2)`）

### Phase 3: CPU Kernels Public Interface の作成

3. **`include/gradflow/autograd/cpu/kernels.hpp` を作成**
   - `CPUKernels` クラスの宣言
   - すべての public メソッドのドキュメント
   - `alignedAlloc`, `alignedFree` ヘルパー関数

### Phase 4: CPU Kernels 実装

4. **`src/autograd/cpu/kernels_avx2.cpp` を作成**
   - CPUID による機能検出（AVX2/AVX-512）
   - 各 elementwise 演算の実装（SIMD またはスカラーフォールバック）
   - `getSIMDInfo()` メソッド
   - アライメント付きメモリ確保/解放

### Phase 5: CMake 統合

5. **CMake ファイルを更新**
   - `src/autograd/cpu/CMakeLists.txt` を作成
   - AVX2 コンパイルフラグ（`-mavx2 -mfma`）
   - OpenMP の検出とリンク
   - ルート `CMakeLists.txt` に `GRADFLOW_ENABLE_CPU_OPTIMIZATIONS` オプション追加

### Phase 6: テストの実装

6. **`tests/test_cpu_optimized.cpp` を作成**
   - `CPUOptimizedTest::AddSIMD_Correctness`
   - `CPUOptimizedTest::MulSIMD_Correctness`
   - `CPUOptimizedTest::BlockedMatMul_Correctness`
   - `CPUOptimizedTest::Performance_Add`
   - `CPUOptimizedTest::Performance_MatMul`
   - `CPUOptimizedTest::Performance_MatMul_vs_Eigen` (optional)

### Phase 7: ドキュメント

7. **README.md を更新**
   - CPU 最適化機能の説明
   - ベンチマーク結果の記載
   - コンパイル方法（AVX2 有効化）

---

## 8. 完了基準

- [ ] すべてのテストが pass（最低 5 個のテスト）
- [ ] 最適化前後で数値的正確性が保たれている（相対誤差 < 1e-5）
- [ ] ベンチマークで最適化前より高速
  - Add: 最適化版が naive 版より 3x 以上高速
  - MatMul: 最適化版が naive 版より 10x 以上高速
- [ ] MatMul が Eigen の 80% 以上の速度
- [ ] AVX2 が使用できない環境でのフォールバック動作確認
- [ ] CI で CPU 最適化テストが実行される

---

## 9. トレードオフと設計判断

### メリット

1. **高パフォーマンス**: SIMD とマルチスレッドにより、大幅な高速化を実現
2. **キャッシュ効率**: Blocked MatMul により、メモリアクセスパターンを最適化
3. **ポータビリティ**: AVX2 が使用できない環境ではスカラー実装にフォールバック
4. **拡張性**: 新しい SIMD 命令セット（AVX-512, ARM NEON）を追加しやすい
5. **ベンチマーク駆動**: すべての最適化は測定可能なパフォーマンス向上を伴う

### リスク・注意点

1. **プラットフォーム依存**: AVX2 は x86_64 専用（ARM では NEON が必要）
2. **アライメント要件**: メモリが 32 バイトアライメントされていることを前提
3. **OpenMP の利用可能性**: OpenMP がない環境では並列化が無効
4. **小さい配列**: 非常に小さい配列では、オーバーヘッドにより最適化効果が限定的
5. **Eigen との比較**: Eigen は高度に最適化されており、80% の目標達成には工夫が必要

---

## 10. パフォーマンス目標

### 目標値

| 演算 | 目標 |
|------|------|
| Add (10M 要素) | 最適化前より 3x 以上高速 |
| MatMul (512x512) | 最適化前より 10x 以上高速 |
| MatMul vs Eigen | Eigen の 80% 以上の速度 |

### ベンチマーク環境

- **CPU**: Intel Core i7 または Apple M1/M2
- **コンパイラ**: GCC 11+ または Clang 14+
- **最適化フラグ**: `-O3 -march=native -mavx2 -mfma`

---

## 11. 参考資料とリンク

### 公式ドキュメント

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [OpenMP 5.2 Specification](https://www.openmp.org/specifications/)
- [GCC Optimization Options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)

### 学術論文・資料

- [Goto, K., & Geijn, R. (2008). Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)
- [Ulrich Drepper. (2007). What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)

### 実装例とチュートリアル

- [UC Berkeley CS267: Matrix Multiplication Optimization](https://people.eecs.berkeley.edu/~demmel/cs267_Spr16/Lectures/lecture11_matmul_jwd16_4pp.pdf)
- [Intel oneAPI: Optimizing BLAS with AVX2](https://www.intel.com/content/www/us/en/developer/articles/technical/optimizing-blas-library-calls-using-avx2.html)

---

以上が Issue #19 の詳細設計書です。この設計に基づき、github-issue-implementer が実装を進めることで、GradFlow に CPU 最適化サポートが追加されます。
