// src/autograd/cpu/matmul_blocked.cpp
#include "gradflow/autograd/cpu/kernels.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cstddef>  // for ptrdiff_t

namespace gradflow {
namespace cpu {

namespace {

// ブロックサイズ（L1 キャッシュに収まるサイズ）
// OpenMP のループ変数（ptrdiff_t）との互換性のため、ptrdiff_t で定義
constexpr ptrdiff_t kBlockSize = 64;

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

    c00 += a0 * b0;
    c01 += a0 * b1;
    c02 += a0 * b2;
    c03 += a0 * b3;
    c10 += a1 * b0;
    c11 += a1 * b1;
    c12 += a1 * b2;
    c13 += a1 * b3;
    c20 += a2 * b0;
    c21 += a2 * b1;
    c22 += a2 * b2;
    c23 += a2 * b3;
    c30 += a3 * b0;
    c31 += a3 * b1;
    c32 += a3 * b2;
    c33 += a3 * b3;
  }

  c[0 * ldc + 0] += c00;
  c[0 * ldc + 1] += c01;
  c[0 * ldc + 2] += c02;
  c[0 * ldc + 3] += c03;
  c[1 * ldc + 0] += c10;
  c[1 * ldc + 1] += c11;
  c[1 * ldc + 2] += c12;
  c[1 * ldc + 3] += c13;
  c[2 * ldc + 0] += c20;
  c[2 * ldc + 1] += c21;
  c[2 * ldc + 2] += c22;
  c[2 * ldc + 3] += c23;
  c[3 * ldc + 0] += c30;
  c[3 * ldc + 1] += c31;
  c[3 * ldc + 2] += c32;
  c[3 * ldc + 3] += c33;
}

// ブロック行列乗算: C_block = A_block * B_block
void matmul_block(const float* a, const float* b, float* c, size_t m, size_t k,
                  size_t n, size_t lda, size_t ldb, size_t ldc) {
  // 4x4 マイクロカーネルを使用
  size_t i = 0;
  for (; i + 4 <= m; i += 4) {
    size_t j = 0;
    for (; j + 4 <= n; j += 4) {
      matmul_kernel_4x4(a + i * lda, b + j, c + i * ldc + j, k, lda, ldb, ldc);
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

void CPUKernels::matmul(const float* a, const float* b, float* c, size_t m,
                        size_t k, size_t n) {
  // 入力検証
  if (a == nullptr || b == nullptr || c == nullptr) {
    return;
  }
  if (m == 0 || k == 0 || n == 0) {
    return;
  }

  // C を 0 で初期化（型安全な方法）
  std::fill_n(c, m * n, 0.0f);

#ifdef _OPENMP
// OpenMP 並列化: ブロックごとに並列実行
// Note: MSVC の OpenMP 2.0 は collapse
// 句をサポートしないため、最も外側のループのみ並列化
#pragma omp parallel for schedule(static)
  for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(m); i += kBlockSize) {
    for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(n); j += kBlockSize) {
      for (ptrdiff_t p = 0; p < static_cast<ptrdiff_t>(k); p += kBlockSize) {
        size_t block_m = std::min(static_cast<size_t>(kBlockSize),
                                  m - static_cast<size_t>(i));
        size_t block_k = std::min(static_cast<size_t>(kBlockSize),
                                  k - static_cast<size_t>(p));
        size_t block_n = std::min(static_cast<size_t>(kBlockSize),
                                  n - static_cast<size_t>(j));

        matmul_block(a + static_cast<size_t>(i) * k + static_cast<size_t>(p),
                     b + static_cast<size_t>(p) * n + static_cast<size_t>(j),
                     c + static_cast<size_t>(i) * n + static_cast<size_t>(j),
                     block_m, block_k, block_n, k, n, n);
      }
    }
  }
#else
  // OpenMP が無効な場合はシリアル実行
  for (size_t i = 0; i < m; i += static_cast<size_t>(kBlockSize)) {
    for (size_t j = 0; j < n; j += static_cast<size_t>(kBlockSize)) {
      for (size_t p = 0; p < k; p += static_cast<size_t>(kBlockSize)) {
        size_t block_m = std::min(static_cast<size_t>(kBlockSize), m - i);
        size_t block_k = std::min(static_cast<size_t>(kBlockSize), k - p);
        size_t block_n = std::min(static_cast<size_t>(kBlockSize), n - j);

        matmul_block(a + i * k + p, b + p * n + j, c + i * n + j, block_m,
                     block_k, block_n, k, n, n);
      }
    }
  }
#endif
}

}  // namespace cpu
}  // namespace gradflow
