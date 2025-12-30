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
 * @brief Softmax activation function (numerically stable version)
 *
 * Applies the softmax function along the specified dimension.
 * Uses the log-sum-exp trick for numerical stability.
 *
 * Forward (numerically stable):
 *   max_x = max(x, axis=dim)
 *   exp_shifted = exp(x - max_x)
 *   y = exp_shifted / sum(exp_shifted, axis=dim)
 *
 * Backward:
 *   ∂L/∂x = y * (∂L/∂y - Σ(∂L/∂y * y))
 *
 * Properties:
 * - Output is a probability distribution (sum to 1)
 * - Numerically stable (prevents overflow/underflow)
 * - Used for multi-class classification
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class SoftmaxOperation : public Operation<T> {
public:
    /**
     * @brief Constructor
     *
     * @param dim Dimension along which to apply softmax (default: -1, last dim)
     */
    explicit SoftmaxOperation(int dim = -1) : dim_(dim) {}

    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("SoftmaxOperation requires exactly 1 input");
        }

        const auto& x = inputs[0];

        // Normalize dim
        int actual_dim = dim_ < 0 ? static_cast<int>(x.ndim()) + dim_ : dim_;
        if (actual_dim < 0 || actual_dim >= static_cast<int>(x.ndim())) {
            throw std::invalid_argument("Invalid dimension for softmax");
        }

        // Numerically stable softmax using log-sum-exp trick
        // Step 1: Subtract max for numerical stability
        auto max_x = max(x, static_cast<size_t>(actual_dim), /*keepdim=*/true);
        auto x_shifted = sub(x, max_x);

        // Step 2: Compute exp(x - max_x)
        auto exp_shifted = exp(x_shifted);

        // Step 3: Compute sum along the specified dimension
        auto sum_exp = sum(exp_shifted, static_cast<size_t>(actual_dim), /*keepdim=*/true);

        // Step 4: Normalize
        auto result = div(exp_shifted, sum_exp);

        // Save output for backward
        this->saveForBackward("output", result);

        return result;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto output = this->getSavedTensor("output");

        // grad_x = y * (grad_output - Σ(grad_output * y))
        // Step 1: Compute grad_output * y
        auto grad_times_output = mul(grad_output, output);

        // Step 2: Sum along the softmax dimension
        int actual_dim = dim_ < 0 ? static_cast<int>(output.ndim()) + dim_ : dim_;
        auto sum_grad_times_output =
            sum(grad_times_output, static_cast<size_t>(actual_dim), /*keepdim=*/true);

        // Step 3: Subtract from grad_output
        auto grad_minus_sum = sub(grad_output, sum_grad_times_output);

        // Step 4: Multiply by output
        auto grad_x = mul(output, grad_minus_sum);

        return {grad_x};
    }

    [[nodiscard]] std::string name() const override { return "SoftmaxOperation"; }

private:
    int dim_;  ///< Dimension along which to apply softmax
};

}  // namespace gradflow
