#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../operation.hpp"
#include "../tensor.hpp"
#include "elementwise.hpp"
#include "op_utils.hpp"

namespace gradflow {

/**
 * @brief Multiplication operation with automatic differentiation
 *
 * Computes element-wise multiplication with broadcasting support.
 *
 * Forward:
 *   z = x * y
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂z * y
 *   ∂L/∂y = ∂L/∂z * x
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class MulOperation : public Operation<T> {
 public:
  Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
    if (inputs.size() != 2) {
      throw std::invalid_argument("MulOperation requires exactly 2 inputs");
    }

    const auto& x = inputs[0];
    const auto& y = inputs[1];

    // Save inputs for backward
    this->saveForBackward("x", x);
    this->saveForBackward("y", y);

    return mul(x, y);
  }

  std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
    auto x = this->getSavedTensor("x");
    auto y = this->getSavedTensor("y");

    // grad_x = grad_output * y
    auto grad_x = mul(grad_output, y);

    // grad_y = grad_output * x
    auto grad_y = mul(grad_output, x);

    // Handle broadcasting
    grad_x = ops::sumToShape(grad_x, x.shape());
    grad_y = ops::sumToShape(grad_y, y.shape());

    return {grad_x, grad_y};
  }

  [[nodiscard]] std::string name() const override { return "MulOperation"; }
};

}  // namespace gradflow
