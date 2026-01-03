// src/autograd/metal/grad_kernels.metal
// Metal Gradient Compute Shaders for GradFlow
// Copyright (c) 2025 GradFlow Project
//
// This file contains Metal Shading Language kernels for
// backward pass (gradient computation) of various operations.

#include <metal_stdlib>
using namespace metal;

// ===================================================================
// Elementwise Operation Gradients
// ===================================================================

/**
 * @brief Mul の x に関する勾配カーネル: grad_x = grad_output * y
 *
 * Thread Group Size: 256
 * Grid Size: (size + 255) / 256
 *
 * @param grad_output 出力の勾配 [[buffer(0)]]
 * @param y 入力 y [[buffer(1)]]
 * @param grad_x 入力 x の勾配 [[buffer(2)]]
 * @param size 要素数 [[buffer(3)]]
 * @param gid スレッドの global position
 */
kernel void mul_grad_x_kernel(device const float* grad_output [[buffer(0)]],
                              device const float* y [[buffer(1)]],
                              device float* grad_x [[buffer(2)]],
                              constant uint& size [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid < size) {
    grad_x[gid] = grad_output[gid] * y[gid];
  }
}

/**
 * @brief Mul の y に関する勾配カーネル: grad_y = grad_output * x
 *
 * Thread Group Size: 256
 * Grid Size: (size + 255) / 256
 *
 * @param grad_output 出力の勾配 [[buffer(0)]]
 * @param x 入力 x [[buffer(1)]]
 * @param grad_y 入力 y の勾配 [[buffer(2)]]
 * @param size 要素数 [[buffer(3)]]
 * @param gid スレッドの global position
 */
kernel void mul_grad_y_kernel(device const float* grad_output [[buffer(0)]],
                              device const float* x [[buffer(1)]],
                              device float* grad_y [[buffer(2)]],
                              constant uint& size [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid < size) {
    grad_y[gid] = grad_output[gid] * x[gid];
  }
}

// ===================================================================
// Activation Function Gradients
// ===================================================================

/**
 * @brief ReLU の勾配カーネル: grad_x = grad_output * (x > 0 ? 1 : 0)
 *
 * Thread Group Size: 256
 * Grid Size: (size + 255) / 256
 *
 * ReLU の勾配は、入力が正の場合は 1、負の場合は 0 となります。
 * これにより、活性化された (x > 0) ニューロンにのみ勾配が伝播します。
 *
 * @param grad_output 出力の勾配 [[buffer(0)]]
 * @param x 入力 x [[buffer(1)]]
 * @param grad_x 入力 x の勾配 [[buffer(2)]]
 * @param size 要素数 [[buffer(3)]]
 * @param gid スレッドの global position
 */
kernel void relu_grad_kernel(device const float* grad_output [[buffer(0)]],
                             device const float* x [[buffer(1)]],
                             device float* grad_x [[buffer(2)]],
                             constant uint& size [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
  if (gid < size) {
    // x > 0 の場合は勾配を通し、x <= 0 の場合は勾配を 0 にする
    // 注: x == 0.0 の場合は勾配を 0 とする（PyTorch の挙動に準拠）
    // 数値安定性: 厳密な 0.0 比較を使用（epsilon は導入していない）
    // 理由: 浮動小数点の丸め誤差は ReLU の判定に実質的な影響を与えない
    float mask = (x[gid] > 0.0F) ? 1.0F : 0.0F;
    grad_x[gid] = grad_output[gid] * mask;
  }
}

// ===================================================================
// Matrix Operation Gradients
// ===================================================================

// MatMul の勾配計算は既存の matmul カーネル (MPS) を再利用します。
// grad_x = grad_output @ y^T -> matmul(grad_output, y_t)
// grad_y = x^T @ grad_output -> matmul(x_t, grad_output)
// そのため、このファイルには MatMul 専用のカーネルは含まれません。
