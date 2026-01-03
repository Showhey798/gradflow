// tests/test_metal_kernels.cpp
// Unit tests for Metal Compute Shader kernels
// Copyright (c) 2025 GradFlow Project

#include <gtest/gtest.h>

#ifdef __APPLE__
#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

#include "gradflow/autograd/metal/allocator.hpp"
#include "gradflow/autograd/metal/device.hpp"
#include "gradflow/autograd/metal/kernels.hpp"

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

// ===================================================================
// Test 1: Add カーネル
// ===================================================================

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

// ===================================================================
// Test 2: Mul カーネル
// ===================================================================

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

// ===================================================================
// Test 3: Sub カーネル
// ===================================================================

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

// ===================================================================
// Test 4: Div カーネル
// ===================================================================

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

// ===================================================================
// Test 5: Sum カーネル
// ===================================================================

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

// ===================================================================
// Test 6: Mean カーネル
// ===================================================================

TEST_F(MetalKernelsTest, MeanKernel) {
  constexpr size_t size = 1000;

  float* input =
      static_cast<float*>(allocator_->allocate(size * sizeof(float)));
  float* output = static_cast<float*>(allocator_->allocate(sizeof(float)));

  // Initialize: input[i] = i * 2
  for (size_t i = 0; i < size; ++i) {
    input[i] = static_cast<float>(i * 2);
  }

  // Expected mean = (0 + 2 + ... + 1998) / 1000 = (size - 1)
  float expected_mean = static_cast<float>(size - 1);

  kernels_->mean(input, output, size);
  kernels_->synchronize();

  EXPECT_NEAR(output[0], expected_mean, expected_mean * 1e-4);

  allocator_->deallocate(output);
  allocator_->deallocate(input);
}

// ===================================================================
// Test 7: MatMul with MPS (小さい行列)
// ===================================================================

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
  // C[2,0] = 7*1 + 8*3 + 9*5 = 7 + 24 + 45 = 76
  // C[2,1] = 7*2 + 8*4 + 9*6 = 14 + 32 + 54 = 100
  // C[3,0] = 10*1 + 11*3 + 12*5 = 10 + 33 + 60 = 103
  // C[3,1] = 10*2 + 11*4 + 12*6 = 20 + 44 + 72 = 136

  EXPECT_FLOAT_EQ(c[0 * n + 0], 22.0f);
  EXPECT_FLOAT_EQ(c[0 * n + 1], 28.0f);
  EXPECT_FLOAT_EQ(c[1 * n + 0], 49.0f);
  EXPECT_FLOAT_EQ(c[1 * n + 1], 64.0f);
  EXPECT_FLOAT_EQ(c[2 * n + 0], 76.0f);
  EXPECT_FLOAT_EQ(c[2 * n + 1], 100.0f);
  EXPECT_FLOAT_EQ(c[3 * n + 0], 103.0f);
  EXPECT_FLOAT_EQ(c[3 * n + 1], 136.0f);

  allocator_->deallocate(c);
  allocator_->deallocate(b);
  allocator_->deallocate(a);
}

// ===================================================================
// Test 8: MatMul with MPS (大きい行列 - パフォーマンステスト)
// ===================================================================

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
  EXPECT_LT(c[0], 10000.0f);

  allocator_->deallocate(c);
  allocator_->deallocate(b);
  allocator_->deallocate(a);
}

// ===================================================================
// Test 9: CPU と GPU の結果一致検証 (Add)
// ===================================================================

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

// ===================================================================
// Test 10: パフォーマンス比較 (GPU vs CPU - large array)
// ===================================================================

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
