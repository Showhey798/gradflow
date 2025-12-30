#pragma once

#include "../operation.hpp"
#include "../tensor.hpp"
#include "matmul.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gradflow {

/**
 * @brief Matrix multiplication operation with automatic differentiation
 *
 * Computes matrix multiplication of two 2D tensors.
 *
 * Forward:
 *   Z = X @ Y  (where X: [M, K], Y: [K, N], Z: [M, N])
 *
 * Backward:
 *   ∂L/∂X = ∂L/∂Z @ Y^T
 *   ∂L/∂Y = X^T @ ∂L/∂Z
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class MatMulOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("MatMulOperation requires exactly 2 inputs");
        }

        const auto& x = inputs[0];
        const auto& y = inputs[1];

        if (x.ndim() != 2 || y.ndim() != 2) {
            throw std::invalid_argument("MatMulOperation requires 2D tensors");
        }

        // Save inputs for backward
        this->saveForBackward("x", x);
        this->saveForBackward("y", y);

        return matmul(x, y);
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto x = this->getSavedTensor("x");
        auto y = this->getSavedTensor("y");

        // grad_x = grad_output @ y^T
        auto y_t = y.transpose(0, 1);
        auto grad_x = matmul(grad_output, y_t);

        // grad_y = x^T @ grad_output
        auto x_t = x.transpose(0, 1);
        auto grad_y = matmul(x_t, grad_output);

        return {grad_x, grad_y};
    }

    [[nodiscard]] std::string name() const override { return "MatMulOperation"; }
};

}  // namespace gradflow
