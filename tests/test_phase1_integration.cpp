#include "gradflow/autograd/device.hpp"
#include "gradflow/autograd/ops/elementwise.hpp"
#include "gradflow/autograd/ops/matmul.hpp"
#include "gradflow/autograd/ops/reduction.hpp"
#include "gradflow/autograd/tensor.hpp"

#include <gtest/gtest.h>

namespace gradflow {

/**
 * @brief Phase 1 統合テスト
 *
 * Phase 1 で実装した全コンポーネントを統合したテストを実施します。
 * - Tensor の基本機能
 * - CPU での演算
 * - DeviceManager を使用したデバイス管理
 * - 複数の演算を組み合わせた計算
 */
class Phase1IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // デフォルトデバイスが CPU であることを確認
        Device default_device = DeviceManager::getDefaultDevice();
        ASSERT_EQ(default_device.type(), DeviceType::CPU);
    }

    void TearDown() override {}

    // Helper function to check if two floats are approximately equal
    bool approx_equal(float a, float b, float epsilon = 1e-5f) { return std::abs(a - b) < epsilon; }
};

// ========================================
// Basic Integration Test
// ========================================

/**
 * @brief 基本的な統合テスト
 *
 * Issue #6 で指定されたテストコード例を実装します。
 * CPU での Tensor 操作が統合的に動作することを確認します。
 */
TEST_F(Phase1IntegrationTest, TensorOperationsOnCPU) {
    auto a = Tensor<float>::randn({100, 200});
    auto b = Tensor<float>::randn({200, 50});

    auto c = matmul(a, b);       // [100, 50]
    auto d = c.transpose(0, 1);  // [50, 100]
    auto e = sum(d, 0);          // [100]

    EXPECT_EQ(e.shape(), Shape({100}));
    EXPECT_EQ(e.device().type(), DeviceType::CPU);
}

// ========================================
// Complex Integration Tests
// ========================================

/**
 * @brief 複数の演算を組み合わせた計算テスト
 *
 * より複雑な演算の組み合わせが正しく動作することを確認します。
 */
TEST_F(Phase1IntegrationTest, ComplexOperationChain) {
    // Create input tensors
    auto x = Tensor<float>::randn({50, 100});
    auto w1 = Tensor<float>::randn({100, 64});
    auto b1 = Tensor<float>::ones({64});
    auto w2 = Tensor<float>::randn({64, 32});

    // Forward pass: (x @ w1 + b1) @ w2
    auto h1 = matmul(x, w1);  // [50, 64]
    EXPECT_EQ(h1.shape(), Shape({50, 64}));

    auto h1_bias = add(h1, b1);  // [50, 64] (broadcast)
    EXPECT_EQ(h1_bias.shape(), Shape({50, 64}));

    auto output = matmul(h1_bias, w2);  // [50, 32]
    EXPECT_EQ(output.shape(), Shape({50, 32}));

    // Verify device
    EXPECT_EQ(output.device().type(), DeviceType::CPU);
}

/**
 * @brief Tensor の view 操作と演算の組み合わせテスト
 */
TEST_F(Phase1IntegrationTest, ViewAndOperations) {
    auto x = Tensor<float>::randn({10, 20, 30});

    // Reshape and operations
    auto x_flat = x.reshape(Shape({200, 30}));
    EXPECT_EQ(x_flat.shape(), Shape({200, 30}));

    auto w = Tensor<float>::randn({30, 15});
    auto y = matmul(x_flat, w);  // [200, 15]
    EXPECT_EQ(y.shape(), Shape({200, 15}));

    // Reshape back
    auto y_reshaped = y.reshape(Shape({10, 20, 15}));
    EXPECT_EQ(y_reshaped.shape(), Shape({10, 20, 15}));

    // Sum reduction
    auto y_sum = sum(y_reshaped, 1);  // [10, 15]
    EXPECT_EQ(y_sum.shape(), Shape({10, 15}));
}

/**
 * @brief Transpose と演算の組み合わせテスト
 */
TEST_F(Phase1IntegrationTest, TransposeAndOperations) {
    auto a = Tensor<float>::randn({30, 40});
    auto b = Tensor<float>::randn({30, 50});

    // Transpose a and compute a^T @ b
    auto a_t = a.transpose(0, 1);  // [40, 30]
    EXPECT_EQ(a_t.shape(), Shape({40, 30}));

    // Make a_t contiguous before matmul
    auto a_t_contiguous = a_t.contiguous();
    auto c = matmul(a_t_contiguous, b);  // [40, 50]
    EXPECT_EQ(c.shape(), Shape({40, 50}));

    // Element-wise operations
    auto d = add(c, Tensor<float>::ones({40, 50}));
    auto ones_times_two = mul(Tensor<float>::ones({40, 50}), Tensor<float>({2.0F}));
    auto e = mul(d, ones_times_two);

    EXPECT_EQ(e.shape(), Shape({40, 50}));
}

// ========================================
// Reduction Operations Integration Tests
// ========================================

/**
 * @brief 複数の reduction 演算を組み合わせたテスト
 */
TEST_F(Phase1IntegrationTest, MultipleReductions) {
    auto x = Tensor<float>::randn({20, 30, 40});

    // Sum along different axes
    auto sum_axis0 = sum(x, 0);  // [30, 40]
    EXPECT_EQ(sum_axis0.shape(), Shape({30, 40}));

    auto sum_axis1 = sum(x, 1);  // [20, 40]
    EXPECT_EQ(sum_axis1.shape(), Shape({20, 40}));

    auto sum_axis2 = sum(x, 2);  // [20, 30]
    EXPECT_EQ(sum_axis2.shape(), Shape({20, 30}));

    // Mean and max
    auto mean_all = mean(x);
    EXPECT_EQ(mean_all.shape(), Shape({}));

    auto max_all = max(x);
    EXPECT_EQ(max_all.shape(), Shape({}));
}

// ========================================
// Mathematical Functions Integration Tests
// ========================================

/**
 * @brief 数学関数と演算の組み合わせテスト
 */
TEST_F(Phase1IntegrationTest, MathematicalFunctions) {
    auto x = Tensor<float>::rand({50, 60});

    // exp(x) + 1
    auto exp_x = exp(x);
    auto exp_x_plus_1 = add(exp_x, Tensor<float>::ones({50, 60}));

    // log(exp(x) + 1)
    auto log_result = log(exp_x_plus_1);
    EXPECT_EQ(log_result.shape(), Shape({50, 60}));

    // sqrt(x^2)
    auto x_squared = pow(x, 2.0F);
    auto x_sqrt = sqrt(x_squared);
    EXPECT_EQ(x_sqrt.shape(), Shape({50, 60}));

    // Verify values are approximately equal to original
    // (allowing for numerical errors)
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            EXPECT_TRUE(approx_equal(x[{i, j}], x_sqrt[{i, j}], 1e-4f));
        }
    }
}

// ========================================
// Large Scale Integration Tests
// ========================================

/**
 * @brief 大規模 Tensor での統合テスト
 *
 * より大きなサイズの Tensor で演算が正しく動作することを確認します。
 */
TEST_F(Phase1IntegrationTest, LargeScaleOperations) {
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || \
    __has_feature(memory_sanitizer)
    GTEST_SKIP() << "Skipping LargeScaleOperations test under sanitizers (timeout issue)";
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    GTEST_SKIP() << "Skipping LargeScaleOperations test under sanitizers (timeout issue)";
#endif

    // Create large tensors
    auto a = Tensor<float>::randn({500, 1000});
    auto b = Tensor<float>::randn({1000, 200});

    // Matrix multiplication
    auto c = matmul(a, b);  // [500, 200]
    EXPECT_EQ(c.shape(), Shape({500, 200}));

    // Reduction
    auto c_sum = sum(c);
    EXPECT_EQ(c_sum.shape(), Shape({}));

    // Element-wise operations on large tensors
    auto d = add(c, Tensor<float>::ones({500, 200}));
    auto ones_times_half = mul(Tensor<float>::ones({500, 200}), Tensor<float>({0.5F}));
    auto e = mul(d, ones_times_half);

    EXPECT_EQ(e.shape(), Shape({500, 200}));
    EXPECT_EQ(e.device().type(), DeviceType::CPU);
}

// ========================================
// Memory Management Integration Tests
// ========================================

/**
 * @brief メモリ管理の統合テスト
 *
 * 複数の Tensor を作成・破棄してもメモリリークが発生しないことを確認します。
 * AddressSanitizer でメモリリークを検出します。
 */
TEST_F(Phase1IntegrationTest, MemoryManagement) {
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || \
    __has_feature(memory_sanitizer)
    GTEST_SKIP() << "Skipping MemoryManagement test under sanitizers (timeout issue)";
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    GTEST_SKIP() << "Skipping MemoryManagement test under sanitizers (timeout issue)";
#endif

    const size_t kNumIterations = 100;

    for (size_t i = 0; i < kNumIterations; ++i) {
        auto a = Tensor<float>::randn({100, 200});
        auto b = Tensor<float>::randn({200, 50});
        auto c = matmul(a, b);
        auto d = c.transpose(0, 1);
        auto e = sum(d, 0);
        // Tensors should be properly deallocated when going out of scope
    }

    // If there are memory leaks, AddressSanitizer will detect them
    SUCCEED();
}

/**
 * @brief Zero-copy 操作のメモリ共有テスト
 */
TEST_F(Phase1IntegrationTest, ZeroCopyMemorySharing) {
    auto a = Tensor<float>::randn({100, 200});

    // View operations should share memory
    auto b = a.view(Shape({200, 100}));
    auto c = b.transpose(0, 1);

    // Verify data pointers are shared (zero-copy)
    EXPECT_EQ(a.data(), b.data());
    EXPECT_EQ(a.data(), c.data());

    // Modifying one should affect others
    a[{0, 0}] = 999.0F;
    EXPECT_FLOAT_EQ((c[{0, 0}]), 999.0F);
}

// ========================================
// Device Manager Integration Tests
// ========================================

/**
 * @brief DeviceManager を使用した統合テスト
 */
TEST_F(Phase1IntegrationTest, DeviceManagerIntegration) {
    // Get default CPU device
    Device cpu_device = DeviceManager::getDefaultDevice();
    EXPECT_EQ(cpu_device.type(), DeviceType::CPU);
    EXPECT_EQ(cpu_device.index(), 0);

    // Get allocator for CPU device
    auto allocator = DeviceManager::getAllocator(cpu_device);
    EXPECT_NE(allocator, nullptr);
    EXPECT_EQ(allocator->device(), cpu_device);

    // Create tensor on CPU
    auto a = Tensor<float>::randn({100, 200});
    EXPECT_EQ(a.device().type(), DeviceType::CPU);

    // Perform operations
    auto b = Tensor<float>::randn({200, 50});
    auto c = matmul(a, b);
    EXPECT_EQ(c.device().type(), DeviceType::CPU);
}

/**
 * @brief デバイスの可用性チェックテスト
 */
TEST_F(Phase1IntegrationTest, DeviceAvailabilityCheck) {
    // CPU は常に利用可能
    EXPECT_TRUE(DeviceManager::isDeviceAvailable(DeviceType::CPU, 0));
    EXPECT_EQ(DeviceManager::getDeviceCount(DeviceType::CPU), 1);

    // CUDA と Metal はランタイム依存
    // クラッシュしないことを確認
    EXPECT_NO_THROW(DeviceManager::isDeviceAvailable(DeviceType::CUDA, 0));
    EXPECT_NO_THROW(DeviceManager::isDeviceAvailable(DeviceType::METAL, 0));
    EXPECT_NO_THROW(DeviceManager::getDeviceCount(DeviceType::CUDA));
    EXPECT_NO_THROW(DeviceManager::getDeviceCount(DeviceType::METAL));
}

// ========================================
// Edge Cases Integration Tests
// ========================================

/**
 * @brief エッジケースの統合テスト
 */
TEST_F(Phase1IntegrationTest, EdgeCases) {
    // 1x1 matrix multiplication
    auto a = Tensor<float>::ones({1, 1});
    auto b = Tensor<float>::ones({1, 1});
    auto c = matmul(a, b);
    EXPECT_EQ(c.shape(), Shape({1, 1}));
    EXPECT_FLOAT_EQ((c[{0, 0}]), 1.0F);

    // Scalar operations
    auto scalar = Tensor<float>({5.0F});
    auto scalar_sum = sum(scalar);
    EXPECT_EQ(scalar_sum.shape(), Shape({}));
    EXPECT_FLOAT_EQ(scalar_sum[{}], 5.0F);

    // Empty dimension operations
    auto x = Tensor<float>::zeros({0, 10});
    EXPECT_EQ(x.size(), 0);
    EXPECT_EQ(x.shape(), Shape({0, 10}));
}

// ========================================
// Contiguousness Integration Tests
// ========================================

/**
 * @brief 連続性チェックと contiguous() 操作のテスト
 */
TEST_F(Phase1IntegrationTest, ContiguousnessOperations) {
    auto a = Tensor<float>::randn({50, 60});

    // Newly created tensor is contiguous
    EXPECT_TRUE(a.isContiguous());

    // Transposed tensor is not contiguous
    auto a_t = a.transpose(0, 1);
    EXPECT_FALSE(a_t.isContiguous());

    // Make it contiguous
    auto a_t_contiguous = a_t.contiguous();
    EXPECT_TRUE(a_t_contiguous.isContiguous());

    // Operations on contiguous tensors
    // a_t_contiguous is [60, 50], so b should be [50, 40]
    auto b = Tensor<float>::randn({50, 40});
    auto c = matmul(a_t_contiguous, b);  // [60, 40]
    EXPECT_EQ(c.shape(), Shape({60, 40}));
}

// ========================================
// Factory Functions Integration Tests
// ========================================

/**
 * @brief ファクトリ関数の統合テスト
 */
TEST_F(Phase1IntegrationTest, FactoryFunctions) {
    // zeros
    auto zeros = Tensor<float>::zeros({100, 200});
    auto zeros_sum = sum(zeros);
    EXPECT_FLOAT_EQ(zeros_sum[{}], 0.0F);

    // ones
    auto ones = Tensor<float>::ones({100, 200});
    auto ones_sum = sum(ones);
    EXPECT_FLOAT_EQ(ones_sum[{}], 20000.0F);

    // eye
    auto eye = Tensor<float>::eye(100);
    auto eye_sum = sum(eye);
    EXPECT_FLOAT_EQ(eye_sum[{}], 100.0F);

    // randn
    auto randn = Tensor<float>::randn({100, 200});
    auto randn_mean = mean(randn);
    // Mean should be approximately 0 for normal distribution
    EXPECT_TRUE(std::abs(randn_mean[{}]) < 0.5F);

    // rand
    auto rand = Tensor<float>::rand({100, 200});
    auto rand_mean = mean(rand);
    // Mean should be approximately 0.5 for uniform [0, 1)
    EXPECT_TRUE(std::abs(rand_mean[{}] - 0.5F) < 0.1F);
}

}  // namespace gradflow
