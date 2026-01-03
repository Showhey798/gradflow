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
   * **メモリレイアウト**:
   * - 入力行列は row-major format で格納されている必要があります
   * - A[i][j] は a[i * k + j] としてアクセスされます
   * - B[i][j] は b[i * n + j] としてアクセスされます
   * - C[i][j] は c[i * n + j] としてアクセスされます
   * - MPSMatrixDescriptor の rowBytes は行の stride (バイト単位) を表します
   *
   * **例**:
   * ```cpp
   * // A (2x3): [[1, 2, 3],
   * //           [4, 5, 6]]
   * float a[] = {1, 2, 3, 4, 5, 6};
   *
   * // B (3x2): [[1, 2],
   * //           [3, 4],
   * //           [5, 6]]
   * float b[] = {1, 2, 3, 4, 5, 6};
   *
   * matmul(a, b, c, 2, 3, 2);
   * // c (2x2): [[22, 28],
   * //           [49, 64]]
   * ```
   *
   * @param a 入力行列 A (GPU メモリ, row-major, size = m * k)
   * @param b 入力行列 B (GPU メモリ, row-major, size = k * n)
   * @param c 出力行列 C (GPU メモリ, row-major, size = m * n)
   * @param m 行列 A の行数
   * @param k 行列 A の列数 / 行列 B の行数
   * @param n 行列 B の列数
   *
   * @note MPSMatrixDescriptor は内部的に column-major を使用しますが、
   *       rowBytes の設定により row-major データを正しく解釈します
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
