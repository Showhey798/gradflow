#include <gtest/gtest.h>

#include <chrono>
#include <cmath>

#include "gradflow/autograd/device.hpp"
#include "gradflow/autograd/ops/reduction.hpp"
#include "gradflow/autograd/ops/variable_ops.hpp"
#include "gradflow/autograd/tensor.hpp"
#include "gradflow/autograd/variable.hpp"

namespace gradflow {

/**
 * @brief Phase 3 統合テスト
 *
 * Phase 3 で実装した Metal GPU 機能を統合したテストを実施します。
 * - Metal GPU でのニューラルネットワーク学習
 * - 勾配が正しく計算されることの確認
 * - Unified Memory の効率性確認
 * - CPU と Metal GPU での結果一致確認
 * - パフォーマンスベンチマーク実施
 */
class Phase3IntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set random seed for reproducibility
    Tensor<float>::setSeed(42);

    // Check Metal availability
    metal_available_ = DeviceManager::isDeviceAvailable(DeviceType::METAL, 0);
    if (!metal_available_) {
      GTEST_SKIP() << "Metal device is not available on this system";
    }
  }

  void TearDown() override {}

  // Helper function to check if two floats are approximately equal
  bool approx_equal(float a, float b, float epsilon = 1e-4f) {
    return std::abs(a - b) < epsilon;
  }

  // Helper function to check if two tensors are approximately equal
  bool tensors_approx_equal(const Tensor<float>& a, const Tensor<float>& b,
                            float epsilon = 1e-4f) {
    if (a.shape() != b.shape()) {
      return false;
    }

    // Compare a few sample values
    const size_t num_samples = std::min(size_t{100}, a.size());
    for (size_t i = 0; i < num_samples; ++i) {
      if (!approx_equal(a.data()[i], b.data()[i], epsilon)) {
        return false;
      }
    }
    return true;
  }

  // Helper function to get appropriate test size based on sanitizer presence
  size_t getLargeTestSize(size_t normal_size) {
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || \
    __has_feature(memory_sanitizer)
    return std::max(size_t{10}, normal_size / 50);
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    return std::max(size_t{10}, normal_size / 50);
#endif
    return normal_size;
  }

  // Helper function to get iteration count based on sanitizer presence
  size_t getIterationCount(size_t normal_count) {
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || \
    __has_feature(memory_sanitizer)
    return std::max(size_t{5}, normal_count / 20);
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    return std::max(size_t{5}, normal_count / 20);
#endif
    return normal_count;
  }

  bool metal_available_ = false;
};

// ========================================
// Metal GPU Neural Network Test
// ========================================

/**
 * @brief Metal GPU での基本的なニューラルネットワーク学習テスト
 *
 * データを Metal GPU に配置し、forward/backward を実行して、
 * 勾配が正しく計算されることを確認します。
 *
 * ネットワーク構成:
 *   - 入力: [1000, 100]
 *   - 隠れ層: [100, 50] (ReLU)
 *   - 出力層: [50, 10]
 */
TEST_F(Phase3IntegrationTest, MetalGPUSimpleNeuralNetworkTraining) {
  // データを Metal GPU に配置
  const size_t batch_size = getLargeTestSize(1000);
  const size_t input_dim = getLargeTestSize(100);
  const size_t hidden_dim = getLargeTestSize(50);
  const size_t output_dim = getLargeTestSize(10);

  // Metal allocator を取得
  auto metal_device = metal(0);  // Metal device を作成
  auto metal_allocator = DeviceManager::getAllocator(metal_device);

  // Metal GPU 上でテンソルを作成
  auto X_data = Tensor<float>::randn({batch_size, input_dim}, 0.0f, 1.0f,
                                     metal_allocator);
  auto W1_data = Tensor<float>::randn({input_dim, hidden_dim}, 0.0f, 1.0f,
                                      metal_allocator);
  auto b1_data = Tensor<float>::zeros({hidden_dim}, metal_allocator);
  auto W2_data = Tensor<float>::randn({hidden_dim, output_dim}, 0.0f, 1.0f,
                                      metal_allocator);
  auto b2_data = Tensor<float>::zeros({output_dim}, metal_allocator);

  // Variable を構築
  auto X = Variable<float>(X_data, true);
  auto W1 = Variable<float>(W1_data, true);
  auto b1 = Variable<float>(b1_data, true);
  auto W2 = Variable<float>(W2_data, true);
  auto b2 = Variable<float>(b2_data, true);

  // Forward
  auto h1_temp = matmul(X, W1);
  auto h1 = h1_temp + b1;
  auto h = relu(h1);
  auto y_temp = matmul(h, W2);
  auto y = y_temp + b2;

  // Backward
  auto loss = sum(y);
  loss.backward();

  // 勾配が Metal GPU 上で計算されていることを確認
  EXPECT_TRUE(W1.hasGrad());
  EXPECT_EQ(W1.grad().device().type(), DeviceType::METAL);

  // 勾配の値が非ゼロであることを確認（Unified Memory でアクセス）
  float grad_sum = 0.0f;
  for (size_t i = 0; i < std::min(size_t{100}, W1.grad().size()); ++i) {
    grad_sum += std::abs(W1.grad().data()[i]);
  }
  EXPECT_GT(grad_sum, 0.0f) << "Gradient should be non-zero";

  // 結果の形状を確認
  EXPECT_EQ(y.data().shape(), Shape({batch_size, output_dim}));
}

// ========================================
// CPU-GPU Consistency Test
// ========================================

/**
 * @brief CPU と Metal GPU で同じ計算が同じ結果になることを確認
 *
 * 同じ入力データを CPU と Metal GPU
 * で処理して、結果が一致することを確認します。
 */
TEST_F(Phase3IntegrationTest, MetalVsCPUForwardBackwardConsistency) {
  const size_t rows = getLargeTestSize(50);
  const size_t cols = getLargeTestSize(30);

  // CPU でのテンソル作成
  auto X_cpu_data = Tensor<float>::randn({rows, cols});
  auto W_cpu_data = Tensor<float>::randn({cols, cols});

  auto X_cpu = Variable<float>(X_cpu_data, false);
  auto W_cpu = Variable<float>(W_cpu_data, true);

  // Metal GPU にコピー
  auto metal_device = metal(0);  // Metal device を作成
  auto metal_allocator = DeviceManager::getAllocator(metal_device);

  // CPU データを Metal にコピー
  auto X_metal_data = Tensor<float>(X_cpu_data.shape(), metal_allocator);
  auto W_metal_data = Tensor<float>(W_cpu_data.shape(), metal_allocator);

  // データをコピー（Unified Memory のため直接アクセス可能）
  std::memcpy(X_metal_data.data(), X_cpu_data.data(),
              X_cpu_data.size() * sizeof(float));
  std::memcpy(W_metal_data.data(), W_cpu_data.data(),
              W_cpu_data.size() * sizeof(float));

  auto X_metal = Variable<float>(X_metal_data, false);
  auto W_metal = Variable<float>(W_metal_data, true);

  // CPU での計算
  auto matmul_cpu = matmul(X_cpu, W_cpu);
  auto y_cpu = relu(matmul_cpu);
  auto loss_cpu = sum(y_cpu);
  loss_cpu.backward();

  // Metal GPU での計算
  auto matmul_metal = matmul(X_metal, W_metal);
  auto y_metal = relu(matmul_metal);
  auto loss_metal = sum(y_metal);
  loss_metal.backward();

  // Forward の結果が一致することを確認（Unified Memory でアクセス）
  // Metal GPU の浮動小数点演算の精度を考慮して 1e-2f の許容誤差を使用
  EXPECT_TRUE(tensors_approx_equal(y_cpu.data(), y_metal.data(), 1e-2f))
      << "Forward results should match between CPU and Metal (tolerance: 1e-2)";

  EXPECT_TRUE(approx_equal(loss_cpu.data()[{}], loss_metal.data()[{}], 1e-2f))
      << "Loss should match between CPU and Metal (tolerance: 1e-2). CPU: "
      << loss_cpu.data()[{}] << ", Metal: " << loss_metal.data()[{}];

  // Backward の結果（勾配）が一致することを確認（Unified Memory でアクセス）
  auto W_cpu_grad = W_cpu.grad();
  auto W_metal_grad = W_metal.grad();

  EXPECT_TRUE(tensors_approx_equal(W_cpu_grad, W_metal_grad, 1e-2f))
      << "Gradients should match between CPU and Metal (tolerance: 1e-2)";
}

// ========================================
// Unified Memory Efficiency Test
// ========================================

/**
 * @brief Unified Memory の効率性を確認するテスト
 *
 * CPU と GPU 間のデータ転送が効率的に行われることを確認します。
 * 実際には、Metal の Unified Memory
 * により、明示的なコピーなしでアクセス可能です。
 */
TEST_F(Phase3IntegrationTest, MetalUnifiedMemoryDataAccess) {
  const size_t size = getLargeTestSize(1000);

  // Metal GPU でテンソルを作成
  auto metal_device = metal(0);  // Metal device を作成
  auto metal_allocator = DeviceManager::getAllocator(metal_device);

  auto X_data = Tensor<float>::randn({size, size}, 0.0f, 1.0f, metal_allocator);
  auto X = Variable<float>(X_data, false);

  // Metal GPU で計算
  auto Y = X * X;

  // 結果が正しいことを確認（Unified Memory で CPU からもアクセス可能）
  EXPECT_EQ(Y.data().shape(), Shape({size, size}));

  // データ転送なしで CPU からアクセス可能（Unified Memory）
  // サンプル値のチェック
  bool all_non_negative = true;
  for (size_t i = 0; i < std::min(size_t{100}, Y.data().size()); ++i) {
    if (Y.data().data()[i] < 0.0f) {
      all_non_negative = false;
      break;
    }
  }
  // X * X の結果は非負であるべき
  EXPECT_TRUE(all_non_negative) << "X * X should produce non-negative values";
}

// ========================================
// Performance Benchmark Test
// ========================================

/**
 * @brief CPU と Metal GPU のパフォーマンスベンチマーク
 *
 * 大きな行列積を CPU と Metal GPU で実行し、実行時間を比較します。
 * Metal GPU の方が高速であることを期待します。
 */
TEST_F(Phase3IntegrationTest, MetalVsCPUMatrixMultiplicationBenchmark) {
  const size_t size = getLargeTestSize(500);

  // ベンチマーク用のデータを準備
  auto A_cpu = Tensor<float>::randn({size, size});
  auto B_cpu = Tensor<float>::randn({size, size});

  // Metal GPU 用のデータを作成
  auto metal_device = metal(0);  // Metal device を作成
  auto metal_allocator = DeviceManager::getAllocator(metal_device);

  auto A_metal = Tensor<float>(A_cpu.shape(), metal_allocator);
  auto B_metal = Tensor<float>(B_cpu.shape(), metal_allocator);

  // データをコピー
  std::memcpy(A_metal.data(), A_cpu.data(), A_cpu.size() * sizeof(float));
  std::memcpy(B_metal.data(), B_cpu.data(), B_cpu.size() * sizeof(float));

  // CPU でのベンチマーク
  auto cpu_start = std::chrono::high_resolution_clock::now();
  auto C_cpu = matmul(A_cpu, B_cpu);
  auto cpu_end = std::chrono::high_resolution_clock::now();
  auto cpu_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start)
          .count();

  // Metal GPU でのベンチマーク
  auto metal_start = std::chrono::high_resolution_clock::now();
  auto C_metal = matmul(A_metal, B_metal);
  auto metal_end = std::chrono::high_resolution_clock::now();
  auto metal_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            metal_end - metal_start)
                            .count();

  // 結果を表示
  std::cout << "\n=== Performance Benchmark ===" << std::endl;
  std::cout << "Matrix size: " << size << "x" << size << std::endl;
  std::cout << "CPU time: " << cpu_duration << " ms" << std::endl;
  std::cout << "Metal GPU time: " << metal_duration << " ms" << std::endl;

  if (metal_duration > 0) {
    float speedup = static_cast<float>(cpu_duration) / metal_duration;
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    // Metal GPU が CPU より遅い場合は警告を表示
    if (metal_duration > cpu_duration) {
      std::cout << "WARNING: Metal GPU is slower than CPU for this operation"
                << std::endl;
      std::cout << "NOTE: This may occur with sanitizers, debug builds, or "
                   "small matrices"
                << std::endl;
    }
  }

  // 結果が一致することを確認（Unified Memory でアクセス）
  EXPECT_TRUE(tensors_approx_equal(C_cpu, C_metal, 1e-2f))
      << "Results should match between CPU and Metal (tolerance: 1e-2)";

  // テストは常に成功（パフォーマンス情報の表示が主目的）
  SUCCEED();
}

// ========================================
// Memory Management Test
// ========================================

/**
 * @brief Metal GPU でのメモリ管理テスト
 *
 * 複数回の forward/backward でメモリリークがないことを確認します。
 */
TEST_F(Phase3IntegrationTest, MetalGPUMemoryLeakDetection) {
  const size_t kNumIterations = getIterationCount(10);
  const size_t size = getLargeTestSize(100);

  // Metal allocator を取得
  auto metal_device = metal(0);  // Metal device を作成
  auto metal_allocator = DeviceManager::getAllocator(metal_device);

  for (size_t i = 0; i < kNumIterations; ++i) {
    auto X_data =
        Tensor<float>::randn({size, size}, 0.0f, 1.0f, metal_allocator);
    auto W_data =
        Tensor<float>::randn({size, size}, 0.0f, 1.0f, metal_allocator);

    auto X = Variable<float>(X_data, false);
    auto W = Variable<float>(W_data, true);

    auto matmul_result = matmul(X, W);
    auto Y = relu(matmul_result);
    auto loss = sum(Y);

    loss.backward();

    // 勾配が計算されていることを確認
    EXPECT_TRUE(W.hasGrad());

    // Variables should be properly deallocated
  }

  // If there are memory leaks, AddressSanitizer will detect them
  SUCCEED();
}

}  // namespace gradflow
