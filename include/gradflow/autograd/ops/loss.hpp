#pragma once

#include "../operation.hpp"
#include "../tensor.hpp"
#include "elementwise.hpp"
#include "reduction.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gradflow {

/**
 * @brief Mean Squared Error (MSE) Loss
 *
 * Computes the mean squared error between predicted and target values.
 * Commonly used for regression tasks.
 *
 * Forward:
 *   L = (1/N) * Σ(predicted - target)²
 *
 * Backward:
 *   ∂L/∂predicted = (2/N) * (predicted - target)
 *   ∂L/∂target = -(2/N) * (predicted - target)
 *
 * Properties:
 * - Always non-negative
 * - Penalizes large errors more heavily (quadratic)
 * - Numerically stable
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class MSELossOperation : public Operation<T> {
public:
    /**
     * @brief Forward pass: compute MSE loss
     *
     * @param inputs Vector of 2 tensors: [predicted, target]
     * @return Scalar loss tensor
     * @throws std::invalid_argument if inputs size is not 2 or shapes don't match
     */
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("MSELossOperation requires exactly 2 inputs");
        }

        const auto& predicted = inputs[0];
        const auto& target = inputs[1];

        if (predicted.shape() != target.shape()) {
            throw std::invalid_argument(
                "MSELossOperation: predicted and target must have the same shape");
        }

        // Compute (predicted - target)²
        auto diff = sub(predicted, target);
        auto squared = mul(diff, diff);

        // Sum all elements
        auto sum_loss = sum(squared);

        // Compute mean
        Tensor<T> loss(Shape{});
        loss.data()[0] = sum_loss.data()[0] / static_cast<T>(predicted.size());

        // Save for backward
        this->saveForBackward("predicted", predicted);
        this->saveForBackward("target", target);

        return loss;
    }

    /**
     * @brief Backward pass: compute gradients
     *
     * @param grad_output Gradient of loss (typically scalar with value 1.0)
     * @return Vector of 2 gradients: [grad_predicted, grad_target]
     */
    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto predicted = this->getSavedTensor("predicted");
        auto target = this->getSavedTensor("target");

        // grad_predicted = (2/N) * (predicted - target) * grad_output
        auto diff = sub(predicted, target);
        T scale = (T(2) / static_cast<T>(predicted.size())) * grad_output.data()[0];

        Tensor<T> grad_predicted(diff.shape());
        for (size_t i = 0; i < diff.size(); ++i) {
            grad_predicted.data()[i] = scale * diff.data()[i];
        }

        // grad_target = -grad_predicted
        Tensor<T> grad_target(diff.shape());
        for (size_t i = 0; i < diff.size(); ++i) {
            grad_target.data()[i] = -grad_predicted.data()[i];
        }

        return {grad_predicted, grad_target};
    }

    [[nodiscard]] std::string name() const override { return "MSELossOperation"; }
};

/**
 * @brief Negative Log Likelihood (NLL) Loss
 *
 * Computes the negative log likelihood loss.
 * Used with log-probabilities (e.g., output of log_softmax).
 *
 * Forward:
 *   L = -(1/N) * Σ log_probs[target_class]
 *
 * Backward:
 *   ∂L/∂log_probs[i] = -(1/N) if i == target_class else 0
 *
 * Properties:
 * - Expects log-probabilities as input
 * - Target is one-hot encoded
 * - Numerically stable when combined with log_softmax
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class NLLLossOperation : public Operation<T> {
public:
    /**
     * @brief Forward pass: compute NLL loss
     *
     * @param inputs Vector of 2 tensors: [log_probs, target]
     *        - log_probs: [batch_size, num_classes] or [N, ...]
     *        - target: [batch_size, num_classes] (one-hot encoded)
     * @return Scalar loss tensor
     * @throws std::invalid_argument if inputs are invalid
     */
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("NLLLossOperation requires exactly 2 inputs");
        }

        const auto& log_probs = inputs[0];
        const auto& target = inputs[1];

        if (log_probs.shape() != target.shape()) {
            throw std::invalid_argument(
                "NLLLossOperation: log_probs and target must have the same shape");
        }

        // -Σ log_probs[target_class]
        auto target_log_probs = mul(log_probs, target);
        auto sum_loss = sum(target_log_probs);

        // Mean over batch (first dimension)
        size_t batch_size = log_probs.shape()[0];
        Tensor<T> loss(Shape{});
        loss.data()[0] = -sum_loss.data()[0] / static_cast<T>(batch_size);

        // Save for backward
        this->saveForBackward("log_probs", log_probs);
        this->saveForBackward("target", target);

        return loss;
    }

    /**
     * @brief Backward pass: compute gradients
     *
     * @param grad_output Gradient of loss (typically scalar with value 1.0)
     * @return Vector of 2 gradients: [grad_log_probs, grad_target]
     */
    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto log_probs = this->getSavedTensor("log_probs");
        auto target = this->getSavedTensor("target");

        size_t batch_size = log_probs.shape()[0];

        // grad_log_probs = -target / batch_size * grad_output
        T scale = -grad_output.data()[0] / static_cast<T>(batch_size);

        Tensor<T> grad_log_probs(log_probs.shape());
        for (size_t i = 0; i < target.size(); ++i) {
            grad_log_probs.data()[i] = scale * target.data()[i];
        }

        // Target has no gradient
        Tensor<T> grad_target(target.shape());
        for (size_t i = 0; i < grad_target.size(); ++i) {
            grad_target.data()[i] = T(0);
        }

        return {grad_log_probs, grad_target};
    }

    [[nodiscard]] std::string name() const override { return "NLLLossOperation"; }
};

/**
 * @brief Cross Entropy Loss
 *
 * Computes the cross entropy loss between logits and target.
 * Combines log_softmax and NLL loss with numerical stability.
 *
 * Forward:
 *   log_probs = log_softmax(logits)  # numerically stable
 *   L = -Σ(target * log_probs)
 *
 * Backward:
 *   ∂L/∂logits = softmax(logits) - target
 *
 * Properties:
 * - Numerically stable (uses log-sum-exp trick)
 * - Widely used for multi-class classification
 * - Equivalent to log_softmax + NLL but more efficient
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class CrossEntropyLossOperation : public Operation<T> {
public:
    /**
     * @brief Forward pass: compute cross entropy loss
     *
     * @param inputs Vector of 2 tensors: [logits, target]
     *        - logits: [batch_size, num_classes]
     *        - target: [batch_size, num_classes] (one-hot encoded)
     * @return Scalar loss tensor
     * @throws std::invalid_argument if inputs are invalid
     */
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("CrossEntropyLossOperation requires exactly 2 inputs");
        }

        const auto& logits = inputs[0];
        const auto& target = inputs[1];

        if (logits.shape() != target.shape()) {
            throw std::invalid_argument(
                "CrossEntropyLossOperation: logits and target must have the same shape");
        }

        if (logits.ndim() != 2) {
            throw std::invalid_argument(
                "CrossEntropyLossOperation: logits must be 2D [batch_size, num_classes]");
        }

        // Numerically stable log_softmax using log-sum-exp trick
        // log_softmax(x) = x - log(sum(exp(x)))
        //                = x - (max(x) + log(sum(exp(x - max(x)))))

        // Step 1: Subtract max for numerical stability
        auto max_logits = max(logits, 1, /*keepdim=*/true);
        auto shifted = sub(logits, max_logits);

        // Step 2: Compute exp(x - max(x))
        auto exp_shifted = exp(shifted);

        // Step 3: Compute sum along class dimension
        auto sum_exp = sum(exp_shifted, 1, /*keepdim=*/true);

        // Step 4: Compute log(sum_exp)
        auto log_sum_exp = log(sum_exp);

        // Step 5: Add back max to get log(sum(exp(x)))
        auto log_sum_exp_shifted = add(max_logits, log_sum_exp);

        // Step 6: log_softmax = x - log_sum_exp_shifted
        auto log_probs = sub(logits, log_sum_exp_shifted);

        // Step 7: -Σ(target * log_probs)
        auto target_log_probs = mul(target, log_probs);
        auto sum_loss = sum(target_log_probs);

        // Mean over batch
        size_t batch_size = logits.shape()[0];
        Tensor<T> loss(Shape{});
        loss.data()[0] = -sum_loss.data()[0] / static_cast<T>(batch_size);

        // Save for backward
        this->saveForBackward("logits", logits);
        this->saveForBackward("target", target);

        return loss;
    }

    /**
     * @brief Backward pass: compute gradients
     *
     * @param grad_output Gradient of loss (typically scalar with value 1.0)
     * @return Vector of 2 gradients: [grad_logits, grad_target]
     */
    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto logits = this->getSavedTensor("logits");
        auto target = this->getSavedTensor("target");

        size_t batch_size = logits.shape()[0];

        // Compute softmax(logits) using the same stable approach
        auto max_logits = max(logits, 1, /*keepdim=*/true);
        auto shifted = sub(logits, max_logits);
        auto exp_shifted = exp(shifted);
        auto sum_exp = sum(exp_shifted, 1, /*keepdim=*/true);
        auto probs = div(exp_shifted, sum_exp);

        // grad_logits = (softmax(logits) - target) / batch_size * grad_output
        auto grad_logits = sub(probs, target);

        T scale = grad_output.data()[0] / static_cast<T>(batch_size);
        for (size_t i = 0; i < grad_logits.size(); ++i) {
            grad_logits.data()[i] *= scale;
        }

        // Target has no gradient
        Tensor<T> grad_target(target.shape());
        for (size_t i = 0; i < grad_target.size(); ++i) {
            grad_target.data()[i] = T(0);
        }

        return {grad_logits, grad_target};
    }

    [[nodiscard]] std::string name() const override { return "CrossEntropyLossOperation"; }
};

/**
 * @brief Binary Cross Entropy (BCE) Loss
 *
 * Computes the binary cross entropy loss.
 * Used for binary classification tasks.
 *
 * Forward:
 *   L = -(1/N) * Σ[target * log(predicted) + (1-target) * log(1-predicted)]
 *
 * Backward:
 *   ∂L/∂predicted = -(1/N) * [target/predicted - (1-target)/(1-predicted)]
 *
 * Properties:
 * - Expects predicted values in [0, 1] (after sigmoid)
 * - Uses epsilon clamping for numerical stability
 * - Target values should be 0 or 1
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class BCELossOperation : public Operation<T> {
public:
    /**
     * @brief Constructor
     *
     * @param eps Epsilon for numerical stability (default: 1e-7)
     */
    explicit BCELossOperation(T eps = static_cast<T>(1e-7)) : eps_(eps) {}

    /**
     * @brief Forward pass: compute BCE loss
     *
     * @param inputs Vector of 2 tensors: [predicted, target]
     *        - predicted: values in [0, 1] (after sigmoid)
     *        - target: binary values (0 or 1)
     * @return Scalar loss tensor
     * @throws std::invalid_argument if inputs are invalid
     */
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("BCELossOperation requires exactly 2 inputs");
        }

        const auto& predicted = inputs[0];
        const auto& target = inputs[1];

        if (predicted.shape() != target.shape()) {
            throw std::invalid_argument(
                "BCELossOperation: predicted and target must have the same shape");
        }

        // Clamp predicted to [eps, 1-eps] for numerical stability
        Tensor<T> clamped_pred(predicted.shape());
        for (size_t i = 0; i < predicted.size(); ++i) {
            clamped_pred.data()[i] = std::max(eps_, std::min(T(1) - eps_, predicted.data()[i]));
        }

        // BCE: -[target * log(pred) + (1-target) * log(1-pred)]
        Tensor<T> loss_elements(predicted.shape());
        for (size_t i = 0; i < predicted.size(); ++i) {
            T pred_val = clamped_pred.data()[i];
            T target_val = target.data()[i];
            loss_elements.data()[i] = -(target_val * std::log(pred_val) +
                                        (T(1) - target_val) * std::log(T(1) - pred_val));
        }

        // Mean over all elements
        auto sum_loss = sum(loss_elements);
        Tensor<T> loss(Shape{});
        loss.data()[0] = sum_loss.data()[0] / static_cast<T>(predicted.size());

        // Save clamped predictions for backward
        this->saveForBackward("predicted", clamped_pred);
        this->saveForBackward("target", target);

        return loss;
    }

    /**
     * @brief Backward pass: compute gradients
     *
     * @param grad_output Gradient of loss (typically scalar with value 1.0)
     * @return Vector of 2 gradients: [grad_predicted, grad_target]
     */
    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto predicted = this->getSavedTensor("predicted");
        auto target = this->getSavedTensor("target");

        // grad_predicted = -(1/N) * [target/pred - (1-target)/(1-pred)] * grad_output
        T scale = -grad_output.data()[0] / static_cast<T>(predicted.size());

        Tensor<T> grad_predicted(predicted.shape());
        for (size_t i = 0; i < predicted.size(); ++i) {
            T pred_val = predicted.data()[i];
            T target_val = target.data()[i];
            grad_predicted.data()[i] =
                scale * (target_val / pred_val - (T(1) - target_val) / (T(1) - pred_val));
        }

        // Target has no gradient
        Tensor<T> grad_target(target.shape());
        for (size_t i = 0; i < grad_target.size(); ++i) {
            grad_target.data()[i] = T(0);
        }

        return {grad_predicted, grad_target};
    }

    [[nodiscard]] std::string name() const override { return "BCELossOperation"; }

private:
    T eps_;  ///< Epsilon for numerical stability
};

}  // namespace gradflow
