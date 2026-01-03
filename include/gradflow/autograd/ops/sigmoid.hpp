#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../operation.hpp"
#include "../tensor.hpp"
#include "elementwise.hpp"

namespace gradflow {

/**
 * @brief Sigmoid activation function
 *
 * Applies the sigmoid function element-wise.
 *
 * Forward:
 *   y = 1 / (1 + exp(-x))
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂y * y * (1 - y)
 *
 * Properties:
 * - Output range: [0, 1]
 * - Non-linear and differentiable
 * - Can suffer from vanishing gradients for large |x|
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class SigmoidOperation : public Operation<T> {
 public:
  Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
    if (inputs.size() != 1) {
      throw std::invalid_argument("SigmoidOperation requires exactly 1 input");
    }

    const auto& x = inputs[0];

    // y = 1 / (1 + exp(-x))
    // Compute directly in a single loop for efficiency
    Tensor<T> result(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
      result.data()[i] = T(1) / (T(1) + std::exp(-x.data()[i]));
    }

    // Save output for backward (more efficient than recomputing)
    this->saveForBackward("output", result);

    return result;
  }

  std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
    auto output = this->getSavedTensor("output");

    // grad_x = grad_output * output * (1 - output)
    Tensor<T> one_minus_output(output.shape());
    for (size_t i = 0; i < output.size(); ++i) {
      one_minus_output.data()[i] = T(1) - output.data()[i];
    }

    auto output_times_one_minus_output = mul(output, one_minus_output);
    auto grad_x = mul(grad_output, output_times_one_minus_output);

    return {grad_x};
  }

  [[nodiscard]] std::string name() const override { return "SigmoidOperation"; }
};

}  // namespace gradflow
