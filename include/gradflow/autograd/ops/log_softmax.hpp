#pragma once

#include "../operation.hpp"
#include "../tensor.hpp"
#include "elementwise.hpp"
#include "reduction.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gradflow {

/**
 * @brief LogSoftmax activation function (numerically stable version)
 *
 * Applies the log softmax function along the specified dimension.
 * Uses the log-sum-exp trick for numerical stability.
 *
 * Forward (numerically stable):
 *   max_x = max(x, axis=dim)
 *   log_sum_exp = max_x + log(sum(exp(x - max_x), axis=dim))
 *   y = x - log_sum_exp
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂y - sum(∂L/∂y, axis=dim) * exp(y)
 *
 * Properties:
 * - Computes log probabilities
 * - Numerically stable (prevents overflow/underflow)
 * - Often used with NLLLoss for classification
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class LogSoftmaxOperation : public Operation<T> {
public:
    /**
     * @brief Constructor
     *
     * @param dim Dimension along which to apply log softmax (default: -1, last dim)
     */
    explicit LogSoftmaxOperation(int dim = -1) : dim_(dim) {}

    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("LogSoftmaxOperation requires exactly 1 input");
        }

        const auto& x = inputs[0];

        // Normalize dim
        int actual_dim = dim_ < 0 ? static_cast<int>(x.ndim()) + dim_ : dim_;
        if (actual_dim < 0 || actual_dim >= static_cast<int>(x.ndim())) {
            throw std::invalid_argument("Invalid dimension for log_softmax");
        }

        // Numerically stable log_softmax using log-sum-exp trick
        // Step 1: Subtract max for numerical stability
        auto max_x = max(x, static_cast<size_t>(actual_dim), /*keepdim=*/true);
        auto x_shifted = sub(x, max_x);

        // Step 2: Compute log(sum(exp(x - max_x)))
        auto exp_shifted = exp(x_shifted);
        auto sum_exp = sum(exp_shifted, static_cast<size_t>(actual_dim), /*keepdim=*/true);
        auto log_sum_exp = log(sum_exp);

        // Step 3: Add back max_x to get log(sum(exp(x)))
        auto log_sum_exp_original = add(max_x, log_sum_exp);

        // Step 4: Subtract from original x
        auto result = sub(x, log_sum_exp_original);

        // Save output for backward
        this->saveForBackward("output", result);

        return result;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto output = this->getSavedTensor("output");

        // grad_x = grad_output - sum(grad_output, axis=dim) * exp(output)
        // Step 1: Sum grad_output along the dimension
        int actual_dim = dim_ < 0 ? static_cast<int>(output.ndim()) + dim_ : dim_;
        auto sum_grad = sum(grad_output, static_cast<size_t>(actual_dim), /*keepdim=*/true);

        // Step 2: Multiply by exp(output) = softmax
        auto exp_output = exp(output);
        auto sum_grad_times_exp = mul(sum_grad, exp_output);

        // Step 3: Subtract from grad_output
        auto grad_x = sub(grad_output, sum_grad_times_exp);

        return {grad_x};
    }

    [[nodiscard]] std::string name() const override { return "LogSoftmaxOperation"; }

private:
    int dim_;  ///< Dimension along which to apply log softmax
};

}  // namespace gradflow
