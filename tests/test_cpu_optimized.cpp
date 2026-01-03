// tests/test_cpu_optimized.cpp
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "gradflow/autograd/cpu/kernels.hpp"

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

// Test 3: Sub SIMD 最適化の正確性
TEST_F(CPUOptimizedTest, SubSIMD_Correctness) {
  constexpr size_t size = 512;

  float* a = static_cast<float*>(alignedAlloc(size * sizeof(float)));
  float* b = static_cast<float*>(alignedAlloc(size * sizeof(float)));
  float* c = static_cast<float*>(alignedAlloc(size * sizeof(float)));

  for (size_t i = 0; i < size; ++i) {
    a[i] = static_cast<float>(i * 3);
    b[i] = static_cast<float>(i);
  }

  kernels_->sub(a, b, c, size);

  for (size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] - b[i]);
  }

  alignedFree(c);
  alignedFree(b);
  alignedFree(a);
}

// Test 4: Div SIMD 最適化の正確性
TEST_F(CPUOptimizedTest, DivSIMD_Correctness) {
  constexpr size_t size = 512;

  float* a = static_cast<float*>(alignedAlloc(size * sizeof(float)));
  float* b = static_cast<float*>(alignedAlloc(size * sizeof(float)));
  float* c = static_cast<float*>(alignedAlloc(size * sizeof(float)));

  for (size_t i = 0; i < size; ++i) {
    a[i] = static_cast<float>(i + 1) * 10.0f;
    b[i] = 2.0f;
  }

  kernels_->div(a, b, c, size);

  for (size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] / b[i]);
  }

  alignedFree(c);
  alignedFree(b);
  alignedFree(a);
}

// Test 5: Blocked MatMul の正確性
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

// Test 6: 最適化前との性能比較 (Add)
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
  std::chrono::duration<double, std::milli> naive_time =
      end_naive - start_naive;

  std::cout << "Add (" << size << " elements):" << std::endl;
  std::cout << "  Optimized: " << opt_time.count() << " ms" << std::endl;
  std::cout << "  Naive: " << naive_time.count() << " ms" << std::endl;
  std::cout << "  Speedup: " << naive_time.count() / opt_time.count() << "x"
            << std::endl;

  // SIMD が有効な場合のみ、最適化版が高速であることを確認
  // スカラー実装同士ではコンパイラ最適化により結果が変わる可能性があるため
  std::string simd_info = kernels_->getSIMDInfo();
  if (simd_info != "Scalar (No SIMD)") {
    EXPECT_LT(opt_time.count(), naive_time.count());
  } else {
    std::cout << "  Note: SIMD not available, performance comparison skipped"
              << std::endl;
  }

  alignedFree(c_naive);
  alignedFree(c_opt);
  alignedFree(b);
  alignedFree(a);
}

// Test 7: MatMul の性能比較
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
  std::chrono::duration<double, std::milli> naive_time =
      end_naive - start_naive;

  std::cout << "MatMul (" << m << "x" << k << " @ " << k << "x" << n
            << "):" << std::endl;
  std::cout << "  Optimized: " << opt_time.count() << " ms" << std::endl;
  std::cout << "  Naive: " << naive_time.count() << " ms" << std::endl;
  std::cout << "  Speedup: " << naive_time.count() / opt_time.count() << "x"
            << std::endl;

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

// Test 8: SIMD 剰余処理のテスト（8 の倍数でないサイズ）
TEST_F(CPUOptimizedTest, AddSIMD_NonMultipleOf8) {
  constexpr size_t size = 1025;  // 8 で割り切れない

  float* a = static_cast<float*>(alignedAlloc(size * sizeof(float)));
  float* b = static_cast<float*>(alignedAlloc(size * sizeof(float)));
  float* c = static_cast<float*>(alignedAlloc(size * sizeof(float)));

  for (size_t i = 0; i < size; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i * 2);
  }

  kernels_->add(a, b, c, size);

  // 特に最後の要素（剰余部分）を確認
  for (size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] + b[i]) << "Failed at index " << i;
  }

  alignedFree(c);
  alignedFree(b);
  alignedFree(a);
}

// Test 9: 非正方行列のテスト
TEST_F(CPUOptimizedTest, BlockedMatMul_NonSquare) {
  constexpr size_t m = 1, k = 1000, n = 1;

  float* a = static_cast<float*>(alignedAlloc(m * k * sizeof(float)));
  float* b = static_cast<float*>(alignedAlloc(k * n * sizeof(float)));
  float* c = static_cast<float*>(alignedAlloc(m * n * sizeof(float)));

  for (size_t i = 0; i < m * k; ++i) {
    a[i] = 1.0f;
  }
  for (size_t i = 0; i < k * n; ++i) {
    b[i] = 2.0f;
  }

  kernels_->matmul(a, b, c, m, k, n);

  // Expected: c[0] = 1*2*1000 = 2000
  EXPECT_FLOAT_EQ(c[0], 2000.0f);

  alignedFree(c);
  alignedFree(b);
  alignedFree(a);
}

// Test 10: ゼロ除算のテスト
TEST_F(CPUOptimizedTest, DivSIMD_DivideByZero) {
  constexpr size_t size = 8;

  float* a = static_cast<float*>(alignedAlloc(size * sizeof(float)));
  float* b = static_cast<float*>(alignedAlloc(size * sizeof(float)));
  float* c = static_cast<float*>(alignedAlloc(size * sizeof(float)));

  for (size_t i = 0; i < size; ++i) {
    a[i] = 10.0f;
    b[i] = 0.0f;  // ゼロ除算
  }

  kernels_->div(a, b, c, size);

  // IEEE 754: 10.0f / 0.0f = +inf
  for (size_t i = 0; i < size; ++i) {
    EXPECT_TRUE(std::isinf(c[i])) << "Expected inf at index " << i;
  }

  alignedFree(c);
  alignedFree(b);
  alignedFree(a);
}
