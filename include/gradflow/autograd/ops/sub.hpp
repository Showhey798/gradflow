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
 * @brief Subtraction operation with automatic differentiation
 *
 * Computes element-wise subtraction with broadcasting support.
 *
 * Forward:
 *   z = x - y
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂z
 *   ∂L/∂y = -∂L/∂z
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class SubOperation : public Operation<T> {
 public:
  Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
    if (inputs.size() != 2) {
      throw std::invalid_argument("SubOperation requires exactly 2 inputs");
    }

    const auto& x = inputs[0];
    const auto& y = inputs[1];

    // Save input shapes for backward
    Tensor<T> x_shape_holder(x.shape());
    Tensor<T> y_shape_holder(y.shape());

    this->saveForBackward("x_shape_holder", x_shape_holder);
    this->saveForBackward("y_shape_holder", y_shape_holder);

    return sub(x, y);
  }

  std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
    // Retrieve input shapes from saved tensors
    auto x_shape = this->getSavedTensor("x_shape_holder").shape();
    auto y_shape = this->getSavedTensor("y_shape_holder").shape();

    // grad_x = grad_output
    // grad_y = -grad_output
    auto grad_x = ops::sumToShape(grad_output, x_shape);

    // Negate grad_output for y
    Tensor<T> neg_grad_output(grad_output.shape());
    for (size_t i = 0; i < grad_output.size(); ++i) {
      neg_grad_output.data()[i] = -grad_output.data()[i];
    }

    auto grad_y = ops::sumToShape(neg_grad_output, y_shape);

    return {grad_x, grad_y};
  }

  [[nodiscard]] std::string name() const override { return "SubOperation"; }
};

}  // namespace gradflow
