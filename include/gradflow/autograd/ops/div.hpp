#pragma once

#include "../operation.hpp"
#include "../tensor.hpp"
#include "elementwise.hpp"
#include "op_utils.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gradflow {

/**
 * @brief Division operation with automatic differentiation
 *
 * Computes element-wise division with broadcasting support.
 *
 * Forward:
 *   z = x / y
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂z / y
 *   ∂L/∂y = -∂L/∂z * x / (y * y)
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class DivOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("DivOperation requires exactly 2 inputs");
        }

        const auto& x = inputs[0];
        const auto& y = inputs[1];

        // Save inputs for backward
        this->saveForBackward("x", x);
        this->saveForBackward("y", y);

        return div(x, y);
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto x = this->getSavedTensor("x");
        auto y = this->getSavedTensor("y");

        // grad_x = grad_output / y
        auto grad_x = div(grad_output, y);

        // grad_y = -grad_output * x / (y * y)
        auto y_squared = mul(y, y);
        auto grad_y = div(mul(grad_output, x), y_squared);

        // Negate grad_y
        for (size_t i = 0; i < grad_y.size(); ++i) {
            grad_y.data()[i] = -grad_y.data()[i];
        }

        // Handle broadcasting
        grad_x = ops::sumToShape(grad_x, x.shape());
        grad_y = ops::sumToShape(grad_y, y.shape());

        return {grad_x, grad_y};
    }

    [[nodiscard]] std::string name() const override { return "DivOperation"; }
};

}  // namespace gradflow
