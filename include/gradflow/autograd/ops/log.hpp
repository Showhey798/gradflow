#pragma once

#include "../operation.hpp"
#include "../tensor.hpp"
#include "elementwise.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gradflow {

/**
 * @brief Natural logarithm operation with automatic differentiation
 *
 * Computes element-wise natural logarithm.
 *
 * Forward:
 *   z = log(x)
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂z / x
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class LogOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("LogOperation requires exactly 1 input");
        }

        const auto& x = inputs[0];

        // Save input for backward
        this->saveForBackward("x", x);

        return log(x);
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto x = this->getSavedTensor("x");

        // grad_x = grad_output / x
        auto grad_x = div(grad_output, x);

        return {grad_x};
    }

    [[nodiscard]] std::string name() const override { return "LogOperation"; }
};

}  // namespace gradflow
