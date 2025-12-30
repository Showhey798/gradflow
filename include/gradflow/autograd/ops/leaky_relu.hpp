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
 * @brief LeakyReLU activation function
 *
 * Applies the leaky rectified linear unit function element-wise.
 *
 * Forward:
 *   y = x if x > 0 else alpha * x
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂y * (1 if x > 0 else alpha)
 *
 * Properties:
 * - Allows small gradient for negative values
 * - Prevents "dying ReLU" problem
 * - Parameterized by alpha (default: 0.01)
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class LeakyReLUOperation : public Operation<T> {
public:
    /**
     * @brief Constructor
     *
     * @param alpha Slope for negative values (default: 0.01)
     */
    explicit LeakyReLUOperation(T alpha = static_cast<T>(0.01)) : alpha_(alpha) {}

    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("LeakyReLUOperation requires exactly 1 input");
        }

        const auto& x = inputs[0];

        // y = x > 0 ? x : alpha * x
        Tensor<T> result(x.shape());
        for (size_t i = 0; i < x.size(); ++i) {
            result.data()[i] = x.data()[i] > T(0) ? x.data()[i] : alpha_ * x.data()[i];
        }

        // Save input for backward
        this->saveForBackward("input", x);

        return result;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto input = this->getSavedTensor("input");

        // grad_x = grad_output * (x > 0 ? 1 : alpha)
        Tensor<T> mask(input.shape());
        for (size_t i = 0; i < input.size(); ++i) {
            mask.data()[i] = input.data()[i] > T(0) ? T(1) : alpha_;
        }

        auto grad_x = mul(grad_output, mask);

        return {grad_x};
    }

    [[nodiscard]] std::string name() const override { return "LeakyReLUOperation"; }

private:
    T alpha_;  ///< Slope for negative values
};

}  // namespace gradflow
