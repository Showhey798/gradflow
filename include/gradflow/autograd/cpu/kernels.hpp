// include/gradflow/autograd/cpu/kernels.hpp
#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace gradflow::cpu {

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
 *   float* a = static_cast<float*>(alignedAlloc(1024 * sizeof(float)));
 *   float* b = static_cast<float*>(alignedAlloc(1024 * sizeof(float)));
 *   float* c = static_cast<float*>(alignedAlloc(1024 * sizeof(float)));
 *
 *   kernels.add(a, b, c, 1024);  // c = a + b (SIMD 最適化)
 *
 *   alignedFree(c);
 *   alignedFree(b);
 *   alignedFree(a);
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
   * @return SIMD 命令セット名（例: "AVX2", "AVX512", "Scalar (No SIMD)"）
   */
  [[nodiscard]] std::string getSIMDInfo() const;

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

}  // namespace gradflow::cpu
