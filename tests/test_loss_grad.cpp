#include "gradflow/autograd/ops/loss.hpp"
#include "gradflow/autograd/ops/op_utils.hpp"

#include <cmath>

#include <gtest/gtest.h>

using namespace gradflow;

class LossGradTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    bool approx_equal(float a, float b, float epsilon = 1e-5F) { return std::abs(a - b) < epsilon; }
};

// ========================================
// MSELoss Tests
// ========================================

TEST_F(LossGradTest, MSELossForward) {
    // Test basic MSE loss calculation
    auto predicted = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto target = Tensor<float>({1.5F, 2.5F, 3.5F});

    auto op = std::make_shared<MSELossOperation<float>>();
    auto loss = op->forward({predicted, target});

    // MSE = (1/N) * Σ(predicted - target)²
    // diff = [-0.5, -0.5, -0.5]
    // squared = [0.25, 0.25, 0.25]
    // sum = 0.75, mean = 0.75 / 3 = 0.25
    EXPECT_EQ(loss.shape(), Shape({}));  // scalar
    EXPECT_NEAR(loss.data()[0], 0.25F, 1e-5F);
}

TEST_F(LossGradTest, MSELossBackward) {
    auto predicted = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto target = Tensor<float>({1.5F, 2.5F, 3.5F});

    auto op = std::make_shared<MSELossOperation<float>>();
    auto loss = op->forward({predicted, target});

    Tensor<float> grad_output(Shape{});
    grad_output.data()[0] = 1.0F;
    auto grads = op->backward(grad_output);

    // grad_predicted = (2/N) * (predicted - target)
    // = (2/3) * [-0.5, -0.5, -0.5]
    // = [-0.333..., -0.333..., -0.333...]
    ASSERT_EQ(grads.size(), 2UL);
    EXPECT_EQ(grads[0].shape(), predicted.shape());

    float expected_grad = (2.0F / 3.0F) * (-0.5F);
    EXPECT_NEAR(grads[0].data()[0], expected_grad, 1e-5F);
    EXPECT_NEAR(grads[0].data()[1], expected_grad, 1e-5F);
    EXPECT_NEAR(grads[0].data()[2], expected_grad, 1e-5F);
}

TEST_F(LossGradTest, MSELoss2D) {
    // Test with 2D tensors (batch of predictions)
    auto predicted = Tensor<float>({{1.0F, 2.0F}, {3.0F, 4.0F}});
    auto target = Tensor<float>({{1.5F, 2.5F}, {3.5F, 4.5F}});

    auto op = std::make_shared<MSELossOperation<float>>();
    auto loss = op->forward({predicted, target});

    // MSE = (1/4) * (0.25 + 0.25 + 0.25 + 0.25) = 0.25
    EXPECT_EQ(loss.shape(), Shape({}));
    EXPECT_NEAR(loss.data()[0], 0.25F, 1e-5F);
}

TEST_F(LossGradTest, MSELossNumericalGradient) {
    auto predicted = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto target = Tensor<float>({1.5F, 2.5F, 3.5F});

    // Check gradient for each element of predicted
    const float epsilon = 1e-4F;
    const float tolerance = 2e-3F;  // Slightly relaxed tolerance for numerical precision
    bool all_correct = true;

    // First, compute analytical gradient
    auto op_analytical = std::make_shared<MSELossOperation<float>>();
    auto loss = op_analytical->forward({predicted, target});

    Tensor<float> grad_output(Shape{});
    grad_output.data()[0] = 1.0F;
    auto grads = op_analytical->backward(grad_output);

    for (size_t i = 0; i < predicted.size(); ++i) {
        // Create perturbed inputs
        Tensor<float> predicted_plus(predicted.shape());
        Tensor<float> predicted_minus(predicted.shape());

        for (size_t j = 0; j < predicted.size(); ++j) {
            predicted_plus.data()[j] = predicted.data()[j];
            predicted_minus.data()[j] = predicted.data()[j];
        }
        predicted_plus.data()[i] += epsilon;
        predicted_minus.data()[i] -= epsilon;

        // Use separate operation instances for numerical gradient
        auto op_plus = std::make_shared<MSELossOperation<float>>();
        auto op_minus = std::make_shared<MSELossOperation<float>>();

        auto loss_plus = op_plus->forward({predicted_plus, target});
        auto loss_minus = op_minus->forward({predicted_minus, target});

        // Numerical gradient
        float numerical_grad = (loss_plus.data()[0] - loss_minus.data()[0]) / (2.0F * epsilon);
        float analytical_grad = grads[0].data()[i];

        // Check relative error
        float abs_diff = std::abs(numerical_grad - analytical_grad);
        float max_val = std::max(std::abs(numerical_grad), std::abs(analytical_grad));
        float relative_error = (max_val < 1e-7F) ? abs_diff : (abs_diff / max_val);

        if (relative_error > tolerance) {
            all_correct = false;
            break;
        }
    }

    EXPECT_TRUE(all_correct);
}

// ========================================
// NLLLoss Tests
// ========================================

TEST_F(LossGradTest, NLLLossForward) {
    // log_probs: [batch_size=2, num_classes=3]
    auto log_probs = Tensor<float>({{-1.0F, -2.0F, -3.0F}, {-0.5F, -1.5F, -2.5F}});

    // target: one-hot encoded [batch_size=2, num_classes=3]
    // Class 0 for first sample, class 1 for second sample
    auto target = Tensor<float>({{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}});

    auto op = std::make_shared<NLLLossOperation<float>>();
    auto loss = op->forward({log_probs, target});

    // NLL = -(1/N) * Σ log_probs[target_class]
    // = -(1/2) * (-1.0 + -1.5) = -(1/2) * (-2.5) = 1.25
    EXPECT_EQ(loss.shape(), Shape({}));
    EXPECT_NEAR(loss.data()[0], 1.25F, 1e-5F);
}

TEST_F(LossGradTest, NLLLossBackward) {
    auto log_probs = Tensor<float>({{-1.0F, -2.0F, -3.0F}, {-0.5F, -1.5F, -2.5F}});
    auto target = Tensor<float>({{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}});

    auto op = std::make_shared<NLLLossOperation<float>>();
    auto loss = op->forward({log_probs, target});

    Tensor<float> grad_output(Shape{});
    grad_output.data()[0] = 1.0F;
    auto grads = op->backward(grad_output);

    // grad = -target / N = -target / 2
    ASSERT_EQ(grads.size(), 2UL);
    EXPECT_EQ(grads[0].shape(), log_probs.shape());

    EXPECT_NEAR(grads[0].data()[0], -0.5F, 1e-5F);  // -1/2
    EXPECT_NEAR(grads[0].data()[1], 0.0F, 1e-5F);   // 0
    EXPECT_NEAR(grads[0].data()[2], 0.0F, 1e-5F);   // 0
    EXPECT_NEAR(grads[0].data()[3], 0.0F, 1e-5F);   // 0
    EXPECT_NEAR(grads[0].data()[4], -0.5F, 1e-5F);  // -1/2
    EXPECT_NEAR(grads[0].data()[5], 0.0F, 1e-5F);   // 0
}

TEST_F(LossGradTest, NLLLossNumericalGradient) {
    auto log_probs = Tensor<float>({{-1.0F, -2.0F}, {-0.5F, -1.5F}});
    auto target = Tensor<float>({{1.0F, 0.0F}, {0.0F, 1.0F}});

    const float epsilon = 1e-4F;
    const float tolerance = 2e-3F;  // Slightly relaxed tolerance
    bool all_correct = true;

    // Compute analytical gradient
    auto op_analytical = std::make_shared<NLLLossOperation<float>>();
    auto loss = op_analytical->forward({log_probs, target});

    Tensor<float> grad_output(Shape{});
    grad_output.data()[0] = 1.0F;
    auto grads = op_analytical->backward(grad_output);

    for (size_t i = 0; i < log_probs.size(); ++i) {
        Tensor<float> log_probs_plus(log_probs.shape());
        Tensor<float> log_probs_minus(log_probs.shape());

        for (size_t j = 0; j < log_probs.size(); ++j) {
            log_probs_plus.data()[j] = log_probs.data()[j];
            log_probs_minus.data()[j] = log_probs.data()[j];
        }
        log_probs_plus.data()[i] += epsilon;
        log_probs_minus.data()[i] -= epsilon;

        // Use separate operation instances
        auto op_plus = std::make_shared<NLLLossOperation<float>>();
        auto op_minus = std::make_shared<NLLLossOperation<float>>();

        auto loss_plus = op_plus->forward({log_probs_plus, target});
        auto loss_minus = op_minus->forward({log_probs_minus, target});

        float numerical_grad = (loss_plus.data()[0] - loss_minus.data()[0]) / (2.0F * epsilon);
        float analytical_grad = grads[0].data()[i];

        float abs_diff = std::abs(numerical_grad - analytical_grad);
        float max_val = std::max(std::abs(numerical_grad), std::abs(analytical_grad));
        float relative_error = (max_val < 1e-7F) ? abs_diff : (abs_diff / max_val);

        if (relative_error > tolerance) {
            all_correct = false;
            break;
        }
    }

    EXPECT_TRUE(all_correct);
}

// ========================================
// CrossEntropyLoss Tests
// ========================================

TEST_F(LossGradTest, CrossEntropyLossForward) {
    // logits: [batch_size=2, num_classes=3]
    auto logits = Tensor<float>({{2.0F, 1.0F, 0.1F}, {0.5F, 2.5F, 1.0F}});

    // target: one-hot encoded
    auto target = Tensor<float>({{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}});

    auto op = std::make_shared<CrossEntropyLossOperation<float>>();
    auto loss = op->forward({logits, target});

    // CrossEntropy = -Σ target * log_softmax(logits)
    // Loss should be positive and finite
    EXPECT_EQ(loss.shape(), Shape({}));
    EXPECT_GT(loss.data()[0], 0.0F);
    EXPECT_TRUE(std::isfinite(loss.data()[0]));
}

TEST_F(LossGradTest, CrossEntropyLossBackward) {
    auto logits = Tensor<float>({{2.0F, 1.0F, 0.1F}, {0.5F, 2.5F, 1.0F}});
    auto target = Tensor<float>({{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}});

    auto op = std::make_shared<CrossEntropyLossOperation<float>>();
    auto loss = op->forward({logits, target});

    Tensor<float> grad_output(Shape{});
    grad_output.data()[0] = 1.0F;
    auto grads = op->backward(grad_output);

    // grad = (softmax(logits) - target) / batch_size
    ASSERT_EQ(grads.size(), 2UL);
    EXPECT_EQ(grads[0].shape(), logits.shape());

    // Gradients should sum to approximately 0 along class dimension
    // (for each sample)
    float sum_grad_sample_0 = grads[0].data()[0] + grads[0].data()[1] + grads[0].data()[2];
    float sum_grad_sample_1 = grads[0].data()[3] + grads[0].data()[4] + grads[0].data()[5];

    EXPECT_NEAR(sum_grad_sample_0, 0.0F, 1e-5F);
    EXPECT_NEAR(sum_grad_sample_1, 0.0F, 1e-5F);
}

TEST_F(LossGradTest, CrossEntropyLossNumericalGradient) {
    auto logits = Tensor<float>({{1.0F, 2.0F}, {0.5F, 1.5F}});
    auto target = Tensor<float>({{1.0F, 0.0F}, {0.0F, 1.0F}});

    auto op = std::make_shared<CrossEntropyLossOperation<float>>();

    // Manual numerical gradient check
    auto loss = op->forward({logits, target});

    Tensor<float> grad_output(Shape{});
    grad_output.data()[0] = 1.0F;
    auto grads = op->backward(grad_output);

    const float epsilon = 1e-4F;
    const float tolerance = 5e-2F;  // CrossEntropy can have larger error
    bool all_correct = true;

    for (size_t i = 0; i < logits.size(); ++i) {
        Tensor<float> logits_plus(logits.shape());
        Tensor<float> logits_minus(logits.shape());

        for (size_t j = 0; j < logits.size(); ++j) {
            logits_plus.data()[j] = logits.data()[j];
            logits_minus.data()[j] = logits.data()[j];
        }
        logits_plus.data()[i] += epsilon;
        logits_minus.data()[i] -= epsilon;

        auto loss_plus = op->forward({logits_plus, target});
        auto loss_minus = op->forward({logits_minus, target});

        float numerical_grad = (loss_plus.data()[0] - loss_minus.data()[0]) / (2.0F * epsilon);
        float analytical_grad = grads[0].data()[i];

        float abs_diff = std::abs(numerical_grad - analytical_grad);
        float max_val = std::max(std::abs(numerical_grad), std::abs(analytical_grad));
        float relative_error = (max_val < 1e-7F) ? abs_diff : (abs_diff / max_val);

        if (relative_error > tolerance) {
            all_correct = false;
            break;
        }
    }

    EXPECT_TRUE(all_correct);
}

TEST_F(LossGradTest, CrossEntropyLossLargeLogits) {
    // Test numerical stability with very large logits
    auto logits = Tensor<float>({{1000.0F, 999.0F, 998.0F}, {500.0F, 501.0F, 499.0F}});
    auto target = Tensor<float>({{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}});

    auto op = std::make_shared<CrossEntropyLossOperation<float>>();
    auto loss = op->forward({logits, target});

    // Loss should be finite (no overflow/underflow)
    EXPECT_TRUE(std::isfinite(loss.data()[0]));

    Tensor<float> grad_output(Shape{});
    grad_output.data()[0] = 1.0F;
    auto grads = op->backward(grad_output);

    // Gradients should be finite
    for (size_t i = 0; i < grads[0].size(); ++i) {
        EXPECT_TRUE(std::isfinite(grads[0].data()[i]));
    }
}

// ========================================
// BCELoss Tests
// ========================================

TEST_F(LossGradTest, BCELossForward) {
    // predicted: probabilities in [0, 1] (after sigmoid)
    auto predicted = Tensor<float>({0.9F, 0.1F, 0.6F, 0.3F});
    auto target = Tensor<float>({1.0F, 0.0F, 1.0F, 0.0F});

    auto op = std::make_shared<BCELossOperation<float>>();
    auto loss = op->forward({predicted, target});

    // BCE = -(1/N) * Σ[target*log(pred) + (1-target)*log(1-pred)]
    EXPECT_EQ(loss.shape(), Shape({}));
    EXPECT_GT(loss.data()[0], 0.0F);
    EXPECT_TRUE(std::isfinite(loss.data()[0]));
}

TEST_F(LossGradTest, BCELossBackward) {
    auto predicted = Tensor<float>({0.9F, 0.1F, 0.6F, 0.3F});
    auto target = Tensor<float>({1.0F, 0.0F, 1.0F, 0.0F});

    auto op = std::make_shared<BCELossOperation<float>>();
    auto loss = op->forward({predicted, target});

    Tensor<float> grad_output(Shape{});
    grad_output.data()[0] = 1.0F;
    auto grads = op->backward(grad_output);

    // grad = -(1/N) * [target/pred - (1-target)/(1-pred)]
    ASSERT_EQ(grads.size(), 2UL);
    EXPECT_EQ(grads[0].shape(), predicted.shape());

    // All gradients should be finite
    for (size_t i = 0; i < grads[0].size(); ++i) {
        EXPECT_TRUE(std::isfinite(grads[0].data()[i]));
    }
}

TEST_F(LossGradTest, BCELossNumericalGradient) {
    auto predicted = Tensor<float>({0.7F, 0.3F, 0.5F});
    auto target = Tensor<float>({1.0F, 0.0F, 1.0F});

    auto op = std::make_shared<BCELossOperation<float>>();

    // Manual numerical gradient check
    auto loss = op->forward({predicted, target});

    Tensor<float> grad_output(Shape{});
    grad_output.data()[0] = 1.0F;
    auto grads = op->backward(grad_output);

    const float epsilon = 1e-4F;
    const float tolerance = 1e-2F;
    bool all_correct = true;

    for (size_t i = 0; i < predicted.size(); ++i) {
        Tensor<float> predicted_plus(predicted.shape());
        Tensor<float> predicted_minus(predicted.shape());

        for (size_t j = 0; j < predicted.size(); ++j) {
            predicted_plus.data()[j] = predicted.data()[j];
            predicted_minus.data()[j] = predicted.data()[j];
        }
        predicted_plus.data()[i] += epsilon;
        predicted_minus.data()[i] -= epsilon;

        auto loss_plus = op->forward({predicted_plus, target});
        auto loss_minus = op->forward({predicted_minus, target});

        float numerical_grad = (loss_plus.data()[0] - loss_minus.data()[0]) / (2.0F * epsilon);
        float analytical_grad = grads[0].data()[i];

        float abs_diff = std::abs(numerical_grad - analytical_grad);
        float max_val = std::max(std::abs(numerical_grad), std::abs(analytical_grad));
        float relative_error = (max_val < 1e-7F) ? abs_diff : (abs_diff / max_val);

        if (relative_error > tolerance) {
            all_correct = false;
            break;
        }
    }

    EXPECT_TRUE(all_correct);
}

TEST_F(LossGradTest, BCELossNumericalStability) {
    // Test with values close to 0 and 1
    auto predicted = Tensor<float>({0.999F, 0.001F, 0.5F});
    auto target = Tensor<float>({1.0F, 0.0F, 1.0F});

    auto op = std::make_shared<BCELossOperation<float>>();
    auto loss = op->forward({predicted, target});

    // Should not produce inf or nan
    EXPECT_TRUE(std::isfinite(loss.data()[0]));

    Tensor<float> grad_output(Shape{});
    grad_output.data()[0] = 1.0F;
    auto grads = op->backward(grad_output);

    for (size_t i = 0; i < grads[0].size(); ++i) {
        EXPECT_TRUE(std::isfinite(grads[0].data()[i]));
    }
}

// ========================================
// Convergence Tests
// ========================================

TEST_F(LossGradTest, MSELossConvergence) {
    // Simple regression: try to fit predicted to target using gradient descent
    auto target = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto predicted = Tensor<float>({0.0F, 0.0F, 0.0F});  // start from zeros

    auto op = std::make_shared<MSELossOperation<float>>();

    float learning_rate = 0.1F;
    float prev_loss = std::numeric_limits<float>::max();

    for (int iter = 0; iter < 50; ++iter) {
        auto loss = op->forward({predicted, target});

        Tensor<float> grad_output(Shape{});
        grad_output.data()[0] = 1.0F;
        auto grads = op->backward(grad_output);

        // Gradient descent step
        for (size_t i = 0; i < predicted.size(); ++i) {
            predicted.data()[i] -= learning_rate * grads[0].data()[i];
        }

        // Loss should decrease (or stay the same in final iterations)
        EXPECT_LE(loss.data()[0], prev_loss + 1e-5F);
        prev_loss = loss.data()[0];
    }

    // Final loss should be very small
    EXPECT_LT(prev_loss, 0.01F);

    // Predicted should be close to target
    for (size_t i = 0; i < predicted.size(); ++i) {
        EXPECT_NEAR(predicted.data()[i], target.data()[i], 0.1F);
    }
}

TEST_F(LossGradTest, CrossEntropyLossConvergence) {
    // Simple classification: try to fit logits to target
    auto target = Tensor<float>({{1.0F, 0.0F}, {0.0F, 1.0F}});
    auto logits = Tensor<float>({{0.0F, 0.0F}, {0.0F, 0.0F}});  // start from zeros

    auto op = std::make_shared<CrossEntropyLossOperation<float>>();

    float learning_rate = 0.5F;
    float prev_loss = std::numeric_limits<float>::max();

    for (int iter = 0; iter < 100; ++iter) {
        auto loss = op->forward({logits, target});

        Tensor<float> grad_output(Shape{});
        grad_output.data()[0] = 1.0F;
        auto grads = op->backward(grad_output);

        // Gradient descent step
        for (size_t i = 0; i < logits.size(); ++i) {
            logits.data()[i] -= learning_rate * grads[0].data()[i];
        }

        // Loss should generally decrease
        if (iter > 0) {
            EXPECT_LE(loss.data()[0], prev_loss + 1e-4F);
        }
        prev_loss = loss.data()[0];
    }

    // Final loss should be relatively small
    EXPECT_LT(prev_loss, 0.5F);
}
