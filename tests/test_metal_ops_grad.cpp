// tests/test_metal_ops_grad.cpp
// Tests for Metal GPU gradient computation
// Copyright (c) 2025 GradFlow Project

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "gradflow/autograd/metal/allocator.hpp"
#include "gradflow/autograd/metal/device.hpp"
#include "gradflow/autograd/metal/grad_kernels.hpp"
#include "gradflow/autograd/metal/kernels.hpp"
#include "gradflow/autograd/tensor.hpp"

using namespace gradflow;
using namespace gradflow::gpu;

class MetalOpsGradTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!MetalDevice::isAvailable()) {
      GTEST_SKIP() << "Metal is not available on this system";
    }

    device = MetalDevice::create();
    allocator = std::make_shared<MetalAllocator>(device.get());
    kernels = std::make_unique<MetalKernels>(device.get());
    grad_kernels = std::make_unique<MetalGradKernels>(device.get());
  }

  void TearDown() override {
    grad_kernels.reset();
    kernels.reset();
    allocator.reset();
    device.reset();
  }

  std::unique_ptr<MetalDevice> device;
  std::shared_ptr<MetalAllocator> allocator;
  std::unique_ptr<MetalKernels> kernels;
  std::unique_ptr<MetalGradKernels> grad_kernels;
};

// ===================================================================
// Helper Functions
// ===================================================================

template <typename T>
bool tensorsApproxEqual(const Tensor<T>& a, const Tensor<T>& b, T tolerance) {
  if (a.shape() != b.shape()) {
    return false;
  }

  for (size_t i = 0; i < a.size(); ++i) {
    T diff = std::abs(a.data()[i] - b.data()[i]);
    T relative_error = diff / (std::abs(a.data()[i]) + 1e-8F);
    if (diff > tolerance && relative_error > tolerance) {
      return false;
    }
  }

  return true;
}

// ===================================================================
// Mul Gradient Tests
// ===================================================================

TEST_F(MetalOpsGradTest, MulGradient) {
  const size_t kSize = 1024;

  // GPU 上にテンソルを作成
  Tensor<float> x(Shape({kSize}), allocator);
  Tensor<float> y(Shape({kSize}), allocator);
  Tensor<float> grad_output(Shape({kSize}), allocator);

  // データを初期化
  for (size_t i = 0; i < kSize; ++i) {
    x.data()[i] = static_cast<float>(i + 1) * 0.01F;
    y.data()[i] = static_cast<float>(i + 1) * 0.02F;
    grad_output.data()[i] = 1.0F;
  }

  // Forward pass: z = x * y
  Tensor<float> z(Shape({kSize}), allocator);
  kernels->mul(x.data(), y.data(), z.data(), kSize);

  // Backward pass: grad_x = grad_output * y, grad_y = grad_output * x
  Tensor<float> grad_x(Shape({kSize}), allocator);
  Tensor<float> grad_y(Shape({kSize}), allocator);
  grad_kernels->mul_grad(grad_output.data(), x.data(), y.data(), grad_x.data(),
                         grad_y.data(), kSize);

  // 結果を検証
  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_NEAR(grad_x.data()[i], y.data()[i], 1e-5F)
        << "Mismatch at index " << i;
    EXPECT_NEAR(grad_y.data()[i], x.data()[i], 1e-5F)
        << "Mismatch at index " << i;
  }
}

// ===================================================================
// ReLU Gradient Tests
// ===================================================================

TEST_F(MetalOpsGradTest, ReLUGradient) {
  const size_t kSize = 1024;

  // GPU 上にテンソルを作成
  Tensor<float> x(Shape({kSize}), allocator);
  Tensor<float> grad_output(Shape({kSize}), allocator);

  // データを初期化 (正負の値を含む)
  for (size_t i = 0; i < kSize; ++i) {
    x.data()[i] = static_cast<float>(i) - 512.0F;
    grad_output.data()[i] = 1.0F;
  }

  // Forward pass: y = relu(x)
  Tensor<float> y(Shape({kSize}), allocator);
  kernels->relu(x.data(), y.data(), kSize);

  // Backward pass: grad_x = grad_output * (x > 0 ? 1 : 0)
  Tensor<float> grad_x(Shape({kSize}), allocator);
  grad_kernels->relu_grad(grad_output.data(), x.data(), grad_x.data(), kSize);

  // 結果を検証
  for (size_t i = 0; i < kSize; ++i) {
    float expected_grad = (x.data()[i] > 0.0F) ? 1.0F : 0.0F;
    EXPECT_NEAR(grad_x.data()[i], expected_grad, 1e-5F)
        << "Mismatch at index " << i;
  }
}

// ===================================================================
// MatMul Gradient Tests
// ===================================================================

TEST_F(MetalOpsGradTest, MatMulGradient) {
  const size_t kM = 2;
  const size_t kK = 3;
  const size_t kN = 2;

  // GPU 上にテンソルを作成
  Tensor<float> x(Shape({kM, kK}), allocator);            // (2, 3)
  Tensor<float> y(Shape({kK, kN}), allocator);            // (3, 2)
  Tensor<float> grad_output(Shape({kM, kN}), allocator);  // (2, 2)

  // データを初期化
  float x_data[] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  float y_data[] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  for (size_t i = 0; i < kM * kK; ++i) {
    x.data()[i] = x_data[i];
  }
  for (size_t i = 0; i < kK * kN; ++i) {
    y.data()[i] = y_data[i];
  }
  for (size_t i = 0; i < kM * kN; ++i) {
    grad_output.data()[i] = 1.0F;
  }

  // Forward pass: z = x @ y
  Tensor<float> z(Shape({kM, kN}), allocator);
  kernels->matmul(x.data(), y.data(), z.data(), kM, kK, kN);

  // Backward pass:
  // grad_x = grad_output @ y^T
  // grad_y = x^T @ grad_output

  // y^T を計算 (3, 2) -> (2, 3)
  Tensor<float> y_t(Shape({kN, kK}), allocator);
  for (size_t i = 0; i < kK; ++i) {
    for (size_t j = 0; j < kN; ++j) {
      y_t.data()[j * kK + i] = y.data()[i * kN + j];
    }
  }

  // x^T を計算 (2, 3) -> (3, 2)
  Tensor<float> x_t(Shape({kK, kM}), allocator);
  for (size_t i = 0; i < kM; ++i) {
    for (size_t j = 0; j < kK; ++j) {
      x_t.data()[j * kM + i] = x.data()[i * kK + j];
    }
  }

  Tensor<float> grad_x(Shape({kM, kK}), allocator);
  Tensor<float> grad_y(Shape({kK, kN}), allocator);

  grad_kernels->matmul_grad_x(grad_output.data(), y_t.data(), grad_x.data(), kM,
                              kK, kN);
  grad_kernels->matmul_grad_y(x_t.data(), grad_output.data(), grad_y.data(), kK,
                              kM, kN);

  // 期待される勾配を計算 (CPU で検証)
  // grad_x = grad_output @ y^T
  // grad_output: (2, 2), y^T: (2, 3) -> grad_x: (2, 3)
  float expected_grad_x[6];
  for (size_t i = 0; i < kM; ++i) {
    for (size_t j = 0; j < kK; ++j) {
      float sum = 0.0F;
      for (size_t l = 0; l < kN; ++l) {
        sum += grad_output.data()[i * kN + l] * y_t.data()[l * kK + j];
      }
      expected_grad_x[i * kK + j] = sum;
    }
  }

  // grad_y = x^T @ grad_output
  // x^T: (3, 2), grad_output: (2, 2) -> grad_y: (3, 2)
  float expected_grad_y[6];
  for (size_t i = 0; i < kK; ++i) {
    for (size_t j = 0; j < kN; ++j) {
      float sum = 0.0F;
      for (size_t l = 0; l < kM; ++l) {
        sum += x_t.data()[i * kM + l] * grad_output.data()[l * kN + j];
      }
      expected_grad_y[i * kN + j] = sum;
    }
  }

  // 結果を検証
  for (size_t i = 0; i < kM * kK; ++i) {
    EXPECT_NEAR(grad_x.data()[i], expected_grad_x[i], 1e-4F)
        << "grad_x mismatch at index " << i;
  }
  for (size_t i = 0; i < kK * kN; ++i) {
    EXPECT_NEAR(grad_y.data()[i], expected_grad_y[i], 1e-4F)
        << "grad_y mismatch at index " << i;
  }
}

// ===================================================================
// Main
// ===================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
