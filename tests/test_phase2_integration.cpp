#include "gradflow/autograd/ops/reduction.hpp"
#include "gradflow/autograd/ops/variable_ops.hpp"
#include "gradflow/autograd/tensor.hpp"
#include "gradflow/autograd/variable.hpp"
#include "gradflow/optim/adam.hpp"

#include <cmath>

#include <gtest/gtest.h>

namespace gradflow {

/**
 * @brief Phase 2 統合テスト
 *
 * Phase 2 で実装した自動微分機能を統合したテストを実施します。
 * - Variable クラスの自動微分
 * - 基本演算の Operation 実装
 * - 活性化関数 (ReLU, Sigmoid)
 * - 損失関数 (MSELoss)
 * - Optimizer (Adam)
 */
class Phase2IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set random seed for reproducibility
        Tensor<float>::setSeed(42);
    }

    void TearDown() override {}

    // Helper function to check if two floats are approximately equal
    bool approx_equal(float a, float b, float epsilon = 1e-5f) { return std::abs(a - b) < epsilon; }

    // Helper function to get appropriate test size based on sanitizer presence
    // Sanitizers add significant overhead, so we use smaller sizes to prevent timeouts
    size_t getLargeTestSize(size_t normal_size) {
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || \
    __has_feature(memory_sanitizer)
        return std::max(size_t{10}, normal_size / 50);  // 50分の1のサイズを使用（最小10）
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
        return std::max(size_t{10}, normal_size / 50);  // 50分の1のサイズを使用（最小10）
#endif
        return normal_size;  // 通常サイズを使用
    }

    // Helper function to get iteration count based on sanitizer presence
    size_t getIterationCount(size_t normal_count) {
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || \
    __has_feature(memory_sanitizer)
        return std::max(size_t{5}, normal_count / 20);  // 20分の1の反復回数を使用（最小5）
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
        return std::max(size_t{5}, normal_count / 20);  // 20分の1の反復回数を使用（最小5）
#endif
        return normal_count;  // 通常の反復回数を使用
    }
};

// ========================================
// XOR Problem Neural Network Test
// ========================================

/**
 * @brief 2 層ニューラルネットワークで XOR 問題を学習するテスト
 *
 * XOR 問題:
 *   Input: [[0, 0], [0, 1], [1, 0], [1, 1]]
 *   Output: [[0], [1], [1], [0]]
 *
 * ネットワーク構成:
 *   - 入力層: 2 次元
 *   - 隠れ層: 4 次元 (ReLU)
 *   - 出力層: 1 次元 (Sigmoid)
 *
 * 完了基準:
 *   - 学習後の予測精度 > 90%
 *   - すべてのサンプルで正解に近い値を出力
 */
TEST_F(Phase2IntegrationTest, SimpleNeuralNetwork) {
    // XOR データ
    auto X = Tensor<float>({{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}});
    // y is [4, 1] shape
    Tensor<float> y(Shape({4, 1}));
    y[{0, 0}] = 0.0f;
    y[{1, 0}] = 1.0f;
    y[{2, 0}] = 1.0f;
    y[{3, 0}] = 0.0f;

    // パラメータ初期化
    auto W1 = Variable<float>(Tensor<float>::randn({2, 4}), true);
    // b1 is [1, 4] for broadcasting compatibility
    auto b1_data = Tensor<float>(Shape({1, 4}));
    for (size_t i = 0; i < 4; ++i) {
        b1_data[{0, i}] = 0.0f;
    }
    auto b1 = Variable<float>(b1_data, true);

    auto W2 = Variable<float>(Tensor<float>::randn({4, 1}), true);
    // b2 is [1, 1] for broadcasting compatibility
    auto b2_data = Tensor<float>(Shape({1, 1}));
    b2_data[{0, 0}] = 0.0f;
    auto b2 = Variable<float>(b2_data, true);

    // Optimizer with higher learning rate for faster convergence
    optim::Adam<float> optimizer(0.1f);
    optimizer.addParamGroup({&W1, &b1, &W2, &b2});

    // 学習ループ
    const size_t kEpochs = getIterationCount(5000);
    for (size_t epoch = 0; epoch < kEpochs; ++epoch) {
        // Forward pass
        auto X_var = Variable<float>(X, false);
        auto h1 = matmul(X_var, W1);
        auto h2 = h1 + b1;
        auto h = relu(h2);
        auto y1 = matmul(h, W2);
        auto y2 = y1 + b2;
        auto y_pred = sigmoid(y2);
        auto y_var = Variable<float>(y, false);
        auto loss = mseLoss(y_pred, y_var);

        // Backward pass
        optimizer.zeroGrad();
        loss.backward();
        optimizer.step();
    }

    // 精度検証
    // 学習が収束していることを確認（最終ロスが初期ロスより小さい）
    auto X_var_final = Variable<float>(X, false);
    auto h1_final = matmul(X_var_final, W1);
    auto h2_final = h1_final + b1;
    auto h_final = relu(h2_final);
    auto y1_final = matmul(h_final, W2);
    auto y2_final = y1_final + b2;
    auto y_pred_final = sigmoid(y2_final);
    auto y_var_final = Variable<float>(y, false);
    auto final_loss = mseLoss(y_pred_final, y_var_final);

    // 最終ロスが 0.1 未満であることを確認（学習が十分に収束している）
    EXPECT_LT(final_loss.data()[{}], 0.1f)
        << "Loss should be less than 0.1. Got: " << final_loss.data()[{}];

    // 各サンプルの予測精度を個別に検証
    const float kThreshold = 0.8f;  // 正解とみなす閾値

    // XOR(0, 0) = 0
    EXPECT_LT((y_pred_final.data()[{0, 0}]), 0.2f)
        << "XOR(0,0) should be close to 0, got " << y_pred_final.data()[{0, 0}];

    // XOR(0, 1) = 1
    EXPECT_GT((y_pred_final.data()[{1, 0}]), kThreshold)
        << "XOR(0,1) should be close to 1, got " << y_pred_final.data()[{1, 0}];

    // XOR(1, 0) = 1
    EXPECT_GT((y_pred_final.data()[{2, 0}]), kThreshold)
        << "XOR(1,0) should be close to 1, got " << y_pred_final.data()[{2, 0}];

    // XOR(1, 1) = 0
    EXPECT_LT((y_pred_final.data()[{3, 0}]), 0.2f)
        << "XOR(1,1) should be close to 0, got " << y_pred_final.data()[{3, 0}];

    // 正解率を計算（デバッグ用）
    int correct = 0;
    for (size_t i = 0; i < 4; ++i) {
        float pred = y_pred_final.data()[{i, 0}];
        float target = y[{i, 0}];
        bool is_correct = (target < 0.5f && pred < 0.5f) || (target > 0.5f && pred > 0.5f);
        if (is_correct) {
            correct++;
        }
    }
    float accuracy = static_cast<float>(correct) / 4.0f;
    EXPECT_GE(accuracy, 1.0f) << "Accuracy: " << (accuracy * 100) << "%";
}

// ========================================
// Gradient Accuracy Test
// ========================================

/**
 * @brief 勾配計算の正確性テスト
 *
 * 簡単な計算グラフで自動微分による勾配が正しいことを確認します。
 *
 * 計算グラフ:
 *   z = sum((x * y) + x)
 *
 * 期待される勾配:
 *   dz/dx = y + 1
 *   dz/dy = x
 */
TEST_F(Phase2IntegrationTest, GradientAccuracy) {
    auto x = Variable<float>(Tensor<float>({2.0f, 3.0f}), true);
    auto y = Variable<float>(Tensor<float>({1.0f, -1.0f}), true);

    // z = sum((x * y) + x)
    // = sum([2*1 + 2, 3*(-1) + 3])
    // = sum([4, 0])
    // = 4
    auto xy = x * y;          // [2, -3]
    auto xy_plus_x = xy + x;  // [4, 0]
    auto z = sum(xy_plus_x);  // 4

    // Backward
    z.backward();

    // dz/dx = y + 1 = [1+1, -1+1] = [2, 0]
    EXPECT_TRUE(x.hasGrad());
    EXPECT_FLOAT_EQ(x.grad()[{0}], 2.0f);
    EXPECT_FLOAT_EQ(x.grad()[{1}], 0.0f);

    // dz/dy = x = [2, 3]
    EXPECT_TRUE(y.hasGrad());
    EXPECT_FLOAT_EQ(y.grad()[{0}], 2.0f);
    EXPECT_FLOAT_EQ(y.grad()[{1}], 3.0f);
}

// ========================================
// Memory Management Test
// ========================================

/**
 * @brief メモリ管理テスト
 *
 * 複数回の forward/backward でメモリリークがないことを確認します。
 * AddressSanitizer で検出されます。
 */
TEST_F(Phase2IntegrationTest, MemoryManagement) {
    const size_t kNumIterations = getIterationCount(10);

    for (size_t i = 0; i < kNumIterations; ++i) {
        // X is [1, 2], W is [2, 1], result is [1, 1]
        Tensor<float> X_data(Shape({1, 2}));
        X_data[{0, 0}] = 1.0f;
        X_data[{0, 1}] = 2.0f;
        auto X = Variable<float>(X_data, false);

        auto W = Variable<float>(Tensor<float>::randn({2, 1}), true);
        auto matmul_result = matmul(X, W);
        auto y_pred = sigmoid(matmul_result);

        Tensor<float> y_true_data(Shape({1, 1}));
        y_true_data[{0, 0}] = 0.5f;
        auto y_true = Variable<float>(y_true_data, false);

        auto loss = mseLoss(y_pred, y_true);

        // Test backward pass to ensure no memory leaks
        loss.backward();

        // Verify that gradient has been computed
        EXPECT_TRUE(W.hasGrad());

        // Variables should be properly deallocated
    }

    // If there are memory leaks, AddressSanitizer will detect them
    SUCCEED();
}

}  // namespace gradflow
