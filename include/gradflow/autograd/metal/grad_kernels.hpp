#pragma once

#ifdef __APPLE__

#include <cstddef>
#include <memory>

namespace gradflow {
namespace gpu {

// Forward declaration
class MetalDevice;
class MetalGradKernelsImpl;

/**
 * @brief Metal GPU での勾配計算カーネル管理クラス
 *
 * 各 Operation の backward pass を Metal GPU 上で実行するための
 * カーネルを提供します。
 *
 * 使用例:
 * @code
 *   auto device = MetalDevice::create();
 *   MetalGradKernels grad_kernels(device.get());
 *
 *   // MatMul の勾配計算
 *   grad_kernels.matmul_grad_x(grad_output, y_t, grad_x, m, k, n);
 *   grad_kernels.matmul_grad_y(x_t, grad_output, grad_y, k, m, n);
 * @endcode
 */
class MetalGradKernels {
 public:
  /**
   * @brief MetalGradKernels を構築
   *
   * @param device Metal デバイス (非 null)
   * @throws std::runtime_error カーネルのロード失敗時
   */
  explicit MetalGradKernels(MetalDevice* device);

  ~MetalGradKernels();

  // コピー・ムーブ禁止
  MetalGradKernels(const MetalGradKernels&) = delete;
  MetalGradKernels& operator=(const MetalGradKernels&) = delete;
  MetalGradKernels(MetalGradKernels&&) = delete;
  MetalGradKernels& operator=(MetalGradKernels&&) = delete;

  // ===== Elementwise Operation Gradients =====

  /**
   * @brief Add の勾配計算: grad_x = grad_output, grad_y = grad_output
   *
   * Add の backward は恒等写像なので、単純なコピーで実装可能。
   * 実際には既存の add_kernel を使用せず、直接メモリコピーで実装。
   *
   * @param grad_output 出力の勾配 (GPU メモリ)
   * @param grad_x 入力 x の勾配 (GPU メモリ)
   * @param grad_y 入力 y の勾配 (GPU メモリ)
   * @param size 要素数
   */
  void add_grad(const float* grad_output, float* grad_x, float* grad_y,
                size_t size);

  /**
   * @brief Mul の勾配計算: grad_x = grad_output * y, grad_y = grad_output * x
   *
   * @param grad_output 出力の勾配 (GPU メモリ)
   * @param x 入力 x (GPU メモリ)
   * @param y 入力 y (GPU メモリ)
   * @param grad_x 入力 x の勾配 (GPU メモリ)
   * @param grad_y 入力 y の勾配 (GPU メモリ)
   * @param size 要素数
   */
  void mul_grad(const float* grad_output, const float* x, const float* y,
                float* grad_x, float* grad_y, size_t size);

  // ===== Activation Function Gradients =====

  /**
   * @brief ReLU の勾配計算: grad_x = grad_output * (x > 0 ? 1 : 0)
   *
   * @param grad_output 出力の勾配 (GPU メモリ)
   * @param x 入力 x (GPU メモリ)
   * @param grad_x 入力 x の勾配 (GPU メモリ)
   * @param size 要素数
   */
  void relu_grad(const float* grad_output, const float* x, float* grad_x,
                 size_t size);

  // ===== Matrix Operation Gradients =====

  /**
   * @brief MatMul の x に関する勾配計算: grad_x = grad_output @ y^T
   *
   * @param grad_output 出力の勾配 (GPU メモリ, row-major, size = m * n)
   * @param y_t 転置された y (GPU メモリ, row-major, size = n * k)
   * @param grad_x 入力 x の勾配 (GPU メモリ, row-major, size = m * k)
   * @param m 行列 grad_output の行数
   * @param k 行列 y_t の列数 (= 行列 x の列数)
   * @param n 行列 grad_output の列数
   */
  void matmul_grad_x(const float* grad_output, const float* y_t, float* grad_x,
                     size_t m, size_t k, size_t n);

  /**
   * @brief MatMul の y に関する勾配計算: grad_y = x^T @ grad_output
   *
   * @param x_t 転置された x (GPU メモリ, row-major, size = k * m)
   * @param grad_output 出力の勾配 (GPU メモリ, row-major, size = m * n)
   * @param grad_y 入力 y の勾配 (GPU メモリ, row-major, size = k * n)
   * @param k 行列 grad_y の行数
   * @param m 行列 x_t の列数
   * @param n 行列 grad_output の列数
   */
  void matmul_grad_y(const float* x_t, const float* grad_output, float* grad_y,
                     size_t k, size_t m, size_t n);

  /**
   * @brief GPU 操作を同期
   *
   * 保留中のすべてのコマンドが完了するまで待機します。
   */
  void synchronize();

 private:
  std::unique_ptr<MetalGradKernelsImpl> impl_;
};

}  // namespace gpu
}  // namespace gradflow

#endif  // __APPLE__
