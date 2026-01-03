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
 * @brief Power operation with automatic differentiation
 *
 * Computes element-wise power with broadcasting support.
 *
 * Forward:
 *   z = x ^ y
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂z * y * x^(y-1)
 *   ∂L/∂y = ∂L/∂z * z * log(x)
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class PowOperation : public Operation<T> {
 public:
  Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
    if (inputs.size() != 2) {
      throw std::invalid_argument("PowOperation requires exactly 2 inputs");
    }

    const auto& x = inputs[0];
    const auto& y = inputs[1];

    // Save inputs for backward
    this->saveForBackward("x", x);
    this->saveForBackward("y", y);

    auto result = pow(x, y);
    this->saveForBackward("output", result);

    return result;
  }

  std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
    auto x = this->getSavedTensor("x");
    auto y = this->getSavedTensor("y");
    auto output = this->getSavedTensor("output");

    // grad_x = grad_output * y * x^(y-1)
    // Create a tensor filled with 1.0
    Tensor<T> one(y.shape());
    for (size_t i = 0; i < one.size(); ++i) {
      one.data()[i] = static_cast<T>(1);
    }

    auto y_minus_1 = sub(y, one);
    auto x_pow_y_minus_1 = pow(x, y_minus_1);
    auto grad_x = mul(mul(grad_output, y), x_pow_y_minus_1);

    // grad_y = grad_output * output * log(x)
    auto log_x = log(x);
    auto grad_y = mul(mul(grad_output, output), log_x);

    // Handle broadcasting
    grad_x = ops::sumToShape(grad_x, x.shape());
    grad_y = ops::sumToShape(grad_y, y.shape());

    return {grad_x, grad_y};
  }

  [[nodiscard]] std::string name() const override { return "PowOperation"; }
};

}  // namespace gradflow
