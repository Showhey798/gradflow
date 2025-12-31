#pragma once

#include "optimizer.hpp"

#include <unordered_map>

#include <gradflow/autograd/tensor.hpp>

namespace gradflow::optim {

/**
 * @brief Stochastic Gradient Descent (SGD) optimizer with momentum
 *
 * Implements the SGD optimization algorithm with optional momentum and weight decay.
 *
 * Update rule without momentum:
 *   w_t = w_t - lr * (g_t + weight_decay * w_t)
 *
 * Update rule with momentum:
 *   v_t = momentum * v_{t-1} + g_t + weight_decay * w_t
 *   w_t = w_t - lr * v_t
 *
 * Where:
 * - w_t: parameter at step t
 * - g_t: gradient at step t
 * - v_t: velocity (momentum buffer) at step t
 * - lr: learning rate
 * - momentum: momentum coefficient (typically 0.9)
 * - weight_decay: L2 penalty coefficient
 *
 * References:
 * - Sutskever et al., "On the importance of initialization and momentum in deep learning" (2013)
 * - Polyak, "Some methods of speeding up the convergence of iteration methods" (1964)
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class SGD : public Optimizer<T> {
public:
    /**
     * @brief Constructs an SGD optimizer
     *
     * @param lr Learning rate (must be positive)
     * @param momentum Momentum factor (default: 0.0, range: [0, 1))
     * @param weight_decay Weight decay (L2 penalty) coefficient (default: 0.0)
     */
    explicit SGD(T lr, T momentum = 0.0, T weight_decay = 0.0)
        : lr_(lr),
          momentum_(momentum),
          weight_decay_(weight_decay) {
        if (lr <= 0) {
            throw std::invalid_argument("Learning rate must be positive");
        }
        if (momentum < 0 || momentum >= 1) {
            throw std::invalid_argument("Momentum must be in [0, 1)");
        }
        if (weight_decay < 0) {
            throw std::invalid_argument("Weight decay must be non-negative");
        }
    }

    /**
     * @brief Performs a single optimization step
     *
     * Updates all parameters using their gradients and the SGD update rule.
     */
    void step() override {
        for (auto* param : this->params_) {
            if (!param->requiresGrad() || !param->hasGrad()) {
                continue;
            }

            // Get parameter data and gradient
            auto& data = param->data();
            const auto& grad = param->grad();

            // Cache pointers for efficiency
            T* data_ptr = data.data();
            const T* grad_ptr = grad.data();
            const size_t data_size = data.size();

            // Apply momentum
            constexpr T kZeroThreshold = T(1e-10);  // Threshold for zero comparison
            if (momentum_ > kZeroThreshold) {
                // Get or create velocity buffer for this parameter
                auto it = velocity_buffers_.find(param);
                if (it == velocity_buffers_.end()) {
                    // Initialize velocity buffer with zeros
                    velocity_buffers_[param] = Tensor<T>::zerosLike(data);
                }

                auto& velocity = velocity_buffers_[param];
                T* velocity_ptr = velocity.data();

                // Update velocity and parameter element-wise
                for (size_t i = 0; i < data_size; ++i) {
                    // Compute effective gradient with weight decay
                    T effective_grad = grad_ptr[i];
                    if (weight_decay_ > kZeroThreshold) {
                        effective_grad += weight_decay_ * data_ptr[i];
                    }

                    // Update velocity: v_t = momentum * v_{t-1} + g_t
                    velocity_ptr[i] = momentum_ * velocity_ptr[i] + effective_grad;

                    // Update parameter: w_t = w_t - lr * v_t
                    data_ptr[i] -= lr_ * velocity_ptr[i];
                }
            } else {
                // Standard SGD without momentum
                for (size_t i = 0; i < data_size; ++i) {
                    // Compute effective gradient with weight decay
                    T effective_grad = grad_ptr[i];
                    if (weight_decay_ > kZeroThreshold) {
                        effective_grad += weight_decay_ * data_ptr[i];
                    }

                    // Update parameter: w_t = w_t - lr * g_t
                    data_ptr[i] -= lr_ * effective_grad;
                }
            }
        }
    }

    /**
     * @brief Returns the learning rate
     */
    [[nodiscard]] T lr() const { return lr_; }

    /**
     * @brief Returns the momentum coefficient
     */
    [[nodiscard]] T momentum() const { return momentum_; }

    /**
     * @brief Returns the weight decay coefficient
     */
    [[nodiscard]] T weightDecay() const { return weight_decay_; }

    /**
     * @brief Sets the learning rate
     */
    void setLr(T lr) {
        if (lr <= 0) {
            throw std::invalid_argument("Learning rate must be positive");
        }
        lr_ = lr;
    }

private:
    T lr_;            ///< Learning rate
    T momentum_;      ///< Momentum coefficient
    T weight_decay_;  ///< Weight decay coefficient

    /// Velocity buffers for momentum (maps parameter pointer to its velocity)
    std::unordered_map<Variable<T>*, Tensor<T>> velocity_buffers_;
};

}  // namespace gradflow::optim
