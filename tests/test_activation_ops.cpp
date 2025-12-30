#include "gradflow/autograd/ops/gelu.hpp"
#include "gradflow/autograd/ops/leaky_relu.hpp"
#include "gradflow/autograd/ops/log_softmax.hpp"
#include "gradflow/autograd/ops/op_utils.hpp"
#include "gradflow/autograd/ops/relu.hpp"
#include "gradflow/autograd/ops/sigmoid.hpp"
#include "gradflow/autograd/ops/softmax.hpp"
#include "gradflow/autograd/ops/tanh_op.hpp"

#include <cmath>

#include <gtest/gtest.h>

using namespace gradflow;

class ActivationOpsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    bool approx_equal(float a, float b, float epsilon = 1e-5F) { return std::abs(a - b) < epsilon; }
};

// ========================================
// ReLU Tests
// ========================================

TEST_F(ActivationOpsTest, ReLUForward) {
    auto x = Tensor<float>({-2.0F, -1.0F, 0.0F, 1.0F, 2.0F});
    auto op = std::make_shared<ReLUOperation<float>>();
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({5}));
    EXPECT_FLOAT_EQ(result[{0}], 0.0F);  // -2 -> 0
    EXPECT_FLOAT_EQ(result[{1}], 0.0F);  // -1 -> 0
    EXPECT_FLOAT_EQ(result[{2}], 0.0F);  //  0 -> 0
    EXPECT_FLOAT_EQ(result[{3}], 1.0F);  //  1 -> 1
    EXPECT_FLOAT_EQ(result[{4}], 2.0F);  //  2 -> 2
}

TEST_F(ActivationOpsTest, ReLUBackward) {
    auto x = Tensor<float>({-2.0F, -1.0F, 0.0F, 1.0F, 2.0F});
    auto op = std::make_shared<ReLUOperation<float>>();
    auto result = op->forward({x});

    auto grad_output = Tensor<float>({1.0F, 1.0F, 1.0F, 1.0F, 1.0F});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 1UL);
    EXPECT_EQ(grads[0].shape(), x.shape());
    EXPECT_FLOAT_EQ(grads[0][{0}], 0.0F);  // x < 0 -> grad = 0
    EXPECT_FLOAT_EQ(grads[0][{1}], 0.0F);  // x < 0 -> grad = 0
    EXPECT_FLOAT_EQ(grads[0][{2}], 0.0F);  // x = 0 -> grad = 0
    EXPECT_FLOAT_EQ(grads[0][{3}], 1.0F);  // x > 0 -> grad = 1
    EXPECT_FLOAT_EQ(grads[0][{4}], 1.0F);  // x > 0 -> grad = 1
}

TEST_F(ActivationOpsTest, ReLU2D) {
    auto x = Tensor<float>({{-1.0F, 2.0F}, {3.0F, -4.0F}});
    auto op = std::make_shared<ReLUOperation<float>>();
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({2, 2}));
    EXPECT_FLOAT_EQ((result[{0, 0}]), 0.0F);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 2.0F);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 3.0F);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 0.0F);
}

TEST_F(ActivationOpsTest, ReLUNumericalGradient) {
    auto x = Tensor<float>({-1.0F, 0.5F, 1.0F});
    auto op = std::make_shared<ReLUOperation<float>>();

    bool grad_correct = ops::test::checkNumericalGradient(*op, {x}, {}, 1e-4F, 1e-2F);
    EXPECT_TRUE(grad_correct);
}

// ========================================
// Sigmoid Tests
// ========================================

TEST_F(ActivationOpsTest, SigmoidForward) {
    auto x = Tensor<float>({-2.0F, 0.0F, 2.0F});
    auto op = std::make_shared<SigmoidOperation<float>>();
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({3}));

    // sigmoid(0) = 0.5
    EXPECT_NEAR(result[{1}], 0.5F, 1e-5F);

    // sigmoid(-x) + sigmoid(x) = 1
    EXPECT_NEAR(result[{0}] + result[{2}], 1.0F, 1e-5F);

    // Output in [0, 1]
    EXPECT_GT(result[{0}], 0.0F);
    EXPECT_LT(result[{0}], 1.0F);
    EXPECT_GT(result[{2}], 0.0F);
    EXPECT_LT(result[{2}], 1.0F);
}

TEST_F(ActivationOpsTest, SigmoidBackward) {
    auto x = Tensor<float>({0.0F});
    auto op = std::make_shared<SigmoidOperation<float>>();
    auto output = op->forward({x});

    // sigmoid(0) = 0.5
    EXPECT_NEAR(output[{0}], 0.5F, 1e-5F);

    auto grad_output = Tensor<float>({1.0F});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 1UL);

    // grad = y * (1 - y) = 0.5 * 0.5 = 0.25
    EXPECT_NEAR(grads[0][{0}], 0.25F, 1e-5F);
}

TEST_F(ActivationOpsTest, SigmoidNumericalGradient) {
    auto x = Tensor<float>({-1.0F, 0.0F, 1.0F});
    auto op = std::make_shared<SigmoidOperation<float>>();

    bool grad_correct = ops::test::checkNumericalGradient(*op, {x}, {}, 1e-4F, 1e-2F);
    EXPECT_TRUE(grad_correct);
}

// ========================================
// Tanh Tests
// ========================================

TEST_F(ActivationOpsTest, TanhForward) {
    auto x = Tensor<float>({-2.0F, 0.0F, 2.0F});
    auto op = std::make_shared<TanhOperation<float>>();
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({3}));

    // tanh(0) = 0
    EXPECT_NEAR(result[{1}], 0.0F, 1e-5F);

    // tanh(-x) = -tanh(x)
    EXPECT_NEAR(result[{0}], -result[{2}], 1e-5F);

    // Output in [-1, 1]
    EXPECT_GT(result[{0}], -1.0F);
    EXPECT_LT(result[{0}], 1.0F);
    EXPECT_GT(result[{2}], -1.0F);
    EXPECT_LT(result[{2}], 1.0F);
}

TEST_F(ActivationOpsTest, TanhBackward) {
    auto x = Tensor<float>({0.0F});
    auto op = std::make_shared<TanhOperation<float>>();
    auto output = op->forward({x});

    // tanh(0) = 0
    EXPECT_NEAR(output[{0}], 0.0F, 1e-5F);

    auto grad_output = Tensor<float>({1.0F});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 1UL);

    // grad = 1 - y² = 1 - 0 = 1
    EXPECT_NEAR(grads[0][{0}], 1.0F, 1e-5F);
}

TEST_F(ActivationOpsTest, TanhNumericalGradient) {
    auto x = Tensor<float>({-1.0F, 0.0F, 1.0F});
    auto op = std::make_shared<TanhOperation<float>>();

    bool grad_correct = ops::test::checkNumericalGradient(*op, {x}, {}, 1e-4F, 1e-2F);
    EXPECT_TRUE(grad_correct);
}

// ========================================
// LeakyReLU Tests
// ========================================

TEST_F(ActivationOpsTest, LeakyReLUForward) {
    float alpha = 0.01F;
    auto x = Tensor<float>({-2.0F, -1.0F, 0.0F, 1.0F, 2.0F});
    auto op = std::make_shared<LeakyReLUOperation<float>>(alpha);
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({5}));
    EXPECT_FLOAT_EQ(result[{0}], -2.0F * alpha);
    EXPECT_FLOAT_EQ(result[{1}], -1.0F * alpha);
    EXPECT_FLOAT_EQ(result[{2}], 0.0F);
    EXPECT_FLOAT_EQ(result[{3}], 1.0F);
    EXPECT_FLOAT_EQ(result[{4}], 2.0F);
}

TEST_F(ActivationOpsTest, LeakyReLUCustomAlpha) {
    float alpha = 0.2F;
    auto x = Tensor<float>({-1.0F, 1.0F});
    auto op = std::make_shared<LeakyReLUOperation<float>>(alpha);
    auto result = op->forward({x});

    EXPECT_FLOAT_EQ(result[{0}], -1.0F * alpha);
    EXPECT_FLOAT_EQ(result[{1}], 1.0F);
}

TEST_F(ActivationOpsTest, LeakyReLUBackward) {
    float alpha = 0.1F;
    auto x = Tensor<float>({-1.0F, 0.0F, 1.0F});
    auto op = std::make_shared<LeakyReLUOperation<float>>(alpha);
    auto result = op->forward({x});

    auto grad_output = Tensor<float>({1.0F, 1.0F, 1.0F});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 1UL);
    EXPECT_FLOAT_EQ(grads[0][{0}], alpha);  // x < 0 -> grad = alpha
    EXPECT_FLOAT_EQ(grads[0][{1}], alpha);  // x = 0 -> grad = alpha
    EXPECT_FLOAT_EQ(grads[0][{2}], 1.0F);   // x > 0 -> grad = 1
}

TEST_F(ActivationOpsTest, LeakyReLUNumericalGradient) {
    float alpha = 0.01F;
    auto x = Tensor<float>({-1.0F, 0.5F, 1.0F});
    auto op = std::make_shared<LeakyReLUOperation<float>>(alpha);

    bool grad_correct = ops::test::checkNumericalGradient(*op, {x}, {}, 1e-4F, 1e-2F);
    EXPECT_TRUE(grad_correct);
}

// ========================================
// GELU Tests
// ========================================

TEST_F(ActivationOpsTest, GELUForward) {
    auto x = Tensor<float>({-3.0F, 0.0F, 3.0F});
    auto op = std::make_shared<GELUOperation<float>>();
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({3}));

    // GELU(0) ≈ 0
    EXPECT_NEAR(result[{1}], 0.0F, 1e-3F);

    // Large positive: GELU(x) ≈ x
    EXPECT_NEAR(result[{2}], 3.0F, 1e-2F);

    // Large negative: GELU(x) ≈ 0
    EXPECT_NEAR(result[{0}], 0.0F, 1e-2F);
}

TEST_F(ActivationOpsTest, GELUBackward) {
    auto x = Tensor<float>({0.0F, 1.0F});
    auto op = std::make_shared<GELUOperation<float>>();
    auto output = op->forward({x});

    auto grad_output = Tensor<float>({1.0F, 1.0F});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 1UL);
    EXPECT_EQ(grads[0].shape(), x.shape());

    // Gradients should be finite
    EXPECT_TRUE(std::isfinite(grads[0][{0}]));
    EXPECT_TRUE(std::isfinite(grads[0][{1}]));
}

TEST_F(ActivationOpsTest, GELUNumericalGradient) {
    auto x = Tensor<float>({-1.0F, 0.0F, 1.0F});
    auto op = std::make_shared<GELUOperation<float>>();

    // GELU is approximate, so use larger tolerance
    bool grad_correct = ops::test::checkNumericalGradient(*op, {x}, {}, 1e-4F, 5e-2F);
    EXPECT_TRUE(grad_correct);
}

// ========================================
// Softmax Tests
// ========================================

TEST_F(ActivationOpsTest, SoftmaxForward) {
    auto x = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto op = std::make_shared<SoftmaxOperation<float>>();
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({3}));

    // Sum should be 1
    float sum = result[{0}] + result[{1}] + result[{2}];
    EXPECT_NEAR(sum, 1.0F, 1e-5F);

    // All elements in [0, 1]
    EXPECT_GT(result[{0}], 0.0F);
    EXPECT_LT(result[{0}], 1.0F);
    EXPECT_GT(result[{1}], 0.0F);
    EXPECT_LT(result[{1}], 1.0F);
    EXPECT_GT(result[{2}], 0.0F);
    EXPECT_LT(result[{2}], 1.0F);

    // Larger input -> larger output
    EXPECT_LT(result[{0}], result[{1}]);
    EXPECT_LT(result[{1}], result[{2}]);
}

TEST_F(ActivationOpsTest, SoftmaxNumericalStabilityLarge) {
    auto x = Tensor<float>({1000.0F, 1001.0F, 1002.0F});
    auto op = std::make_shared<SoftmaxOperation<float>>();
    auto result = op->forward({x});

    // All elements should be finite
    EXPECT_TRUE(std::isfinite(result[{0}]));
    EXPECT_TRUE(std::isfinite(result[{1}]));
    EXPECT_TRUE(std::isfinite(result[{2}]));

    // Sum should be 1
    float sum = result[{0}] + result[{1}] + result[{2}];
    EXPECT_NEAR(sum, 1.0F, 1e-5F);
}

TEST_F(ActivationOpsTest, SoftmaxNumericalStabilitySmall) {
    auto x = Tensor<float>({-1000.0F, -1001.0F, -1002.0F});
    auto op = std::make_shared<SoftmaxOperation<float>>();
    auto result = op->forward({x});

    // All elements should be finite
    EXPECT_TRUE(std::isfinite(result[{0}]));
    EXPECT_TRUE(std::isfinite(result[{1}]));
    EXPECT_TRUE(std::isfinite(result[{2}]));

    // Sum should be 1
    float sum = result[{0}] + result[{1}] + result[{2}];
    EXPECT_NEAR(sum, 1.0F, 1e-5F);
}

TEST_F(ActivationOpsTest, SoftmaxBackward) {
    auto x = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto op = std::make_shared<SoftmaxOperation<float>>();
    auto output = op->forward({x});

    auto grad_output = Tensor<float>({1.0F, 0.0F, 0.0F});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 1UL);
    EXPECT_EQ(grads[0].shape(), x.shape());

    // Gradient sum should be 0 (Softmax property)
    float grad_sum = grads[0][{0}] + grads[0][{1}] + grads[0][{2}];
    EXPECT_NEAR(grad_sum, 0.0F, 1e-5F);
}

TEST_F(ActivationOpsTest, SoftmaxNumericalGradient) {
    auto x = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto op = std::make_shared<SoftmaxOperation<float>>();

    bool grad_correct = ops::test::checkNumericalGradient(*op, {x}, {}, 1e-4F, 1e-2F);
    EXPECT_TRUE(grad_correct);
}

// ========================================
// LogSoftmax Tests
// ========================================

TEST_F(ActivationOpsTest, LogSoftmaxForward) {
    auto x = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto op = std::make_shared<LogSoftmaxOperation<float>>();
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({3}));

    // All elements should be negative (log probabilities)
    EXPECT_LT(result[{0}], 0.0F);
    EXPECT_LT(result[{1}], 0.0F);
    EXPECT_LT(result[{2}], 0.0F);

    // exp(log_softmax) = softmax
    auto softmax_op = std::make_shared<SoftmaxOperation<float>>();
    auto softmax_result = softmax_op->forward({x});

    EXPECT_NEAR(std::exp(result[{0}]), softmax_result[{0}], 1e-5F);
    EXPECT_NEAR(std::exp(result[{1}]), softmax_result[{1}], 1e-5F);
    EXPECT_NEAR(std::exp(result[{2}]), softmax_result[{2}], 1e-5F);
}

TEST_F(ActivationOpsTest, LogSoftmaxNumericalStability) {
    auto x_large = Tensor<float>({1000.0F, 1001.0F, 1002.0F});
    auto op_large = std::make_shared<LogSoftmaxOperation<float>>();
    auto result_large = op_large->forward({x_large});

    // All elements should be finite
    EXPECT_TRUE(std::isfinite(result_large[{0}]));
    EXPECT_TRUE(std::isfinite(result_large[{1}]));
    EXPECT_TRUE(std::isfinite(result_large[{2}]));

    auto x_small = Tensor<float>({-1000.0F, -1001.0F, -1002.0F});
    auto op_small = std::make_shared<LogSoftmaxOperation<float>>();
    auto result_small = op_small->forward({x_small});

    EXPECT_TRUE(std::isfinite(result_small[{0}]));
    EXPECT_TRUE(std::isfinite(result_small[{1}]));
    EXPECT_TRUE(std::isfinite(result_small[{2}]));
}

TEST_F(ActivationOpsTest, LogSoftmaxBackward) {
    auto x = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto op = std::make_shared<LogSoftmaxOperation<float>>();
    auto output = op->forward({x});

    auto grad_output = Tensor<float>({1.0F, 0.0F, 0.0F});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 1UL);
    EXPECT_EQ(grads[0].shape(), x.shape());

    // Gradients should be finite
    EXPECT_TRUE(std::isfinite(grads[0][{0}]));
    EXPECT_TRUE(std::isfinite(grads[0][{1}]));
    EXPECT_TRUE(std::isfinite(grads[0][{2}]));
}

TEST_F(ActivationOpsTest, LogSoftmaxNumericalGradient) {
    auto x = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto op = std::make_shared<LogSoftmaxOperation<float>>();

    bool grad_correct = ops::test::checkNumericalGradient(*op, {x}, {}, 1e-4F, 1e-2F);
    EXPECT_TRUE(grad_correct);
}
