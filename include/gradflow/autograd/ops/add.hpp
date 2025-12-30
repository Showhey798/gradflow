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
 * @brief Addition operation with automatic differentiation
 *
 * Computes element-wise addition with broadcasting support.
 *
 * Forward:
 *   z = x + y
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂z
 *   ∂L/∂y = ∂L/∂z
 *
 * When broadcasting occurs, gradients are summed over the broadcasted dimensions.
 *
 * Example:
 * @code
 * auto x = Tensor<float>({1.0F, 2.0F});
 * auto y = Tensor<float>({3.0F, 4.0F});
 * auto op = std::make_shared<AddOperation<float>>();
 * auto z = op->forward({x, y});  // z = [4.0, 6.0]
 * auto grads = op->backward(Tensor<float>({1.0F, 1.0F}));
 * // grads[0] = [1.0, 1.0], grads[1] = [1.0, 1.0]
 * @endcode
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class AddOperation : public Operation<T> {
public:
    /**
     * @brief Forward pass: compute z = x + y
     *
     * @param inputs Vector containing exactly 2 tensors [x, y]
     * @return Output tensor z
     * @throws std::invalid_argument if inputs.size() != 2
     * @throws std::invalid_argument if shapes are not broadcastable
     */
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("AddOperation requires exactly 2 inputs");
        }

        const auto& x = inputs[0];
        const auto& y = inputs[1];

        // Save input shapes for backward (need to handle broadcasting)
        // Store shapes as dummy tensors with shape information
        Tensor<T> x_shape_holder(x.shape());
        Tensor<T> y_shape_holder(y.shape());

        this->saveForBackward("x_shape_holder", x_shape_holder);
        this->saveForBackward("y_shape_holder", y_shape_holder);

        return add(x, y);
    }

    /**
     * @brief Backward pass: compute gradients
     *
     * @param grad_output Gradient of loss with respect to output (∂L/∂z)
     * @return Vector of gradients [∂L/∂x, ∂L/∂y]
     */
    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        // Retrieve input shapes from saved tensors
        auto x_shape = this->getSavedTensor("x_shape_holder").shape();
        auto y_shape = this->getSavedTensor("y_shape_holder").shape();

        // For addition, gradients flow through unchanged
        // But we need to handle broadcasting
        auto grad_x = ops::sumToShape(grad_output, x_shape);
        auto grad_y = ops::sumToShape(grad_output, y_shape);

        return {grad_x, grad_y};
    }

    /**
     * @brief Returns the name of this operation
     * @return "AddOperation"
     */
    [[nodiscard]] std::string name() const override { return "AddOperation"; }
};

}  // namespace gradflow
