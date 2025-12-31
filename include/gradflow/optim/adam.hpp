#pragma once

#include "optimizer.hpp"

#include <cmath>
#include <unordered_map>

#include <gradflow/autograd/tensor.hpp>

namespace gradflow {
namespace optim {

/**
 * @brief Adam optimizer with optional AdamW weight decay
 *
 * Implements the Adam (Adaptive Moment Estimation) optimization algorithm
 * with support for AdamW-style weight decay.
 *
 * Standard Adam update rule:
 *   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
 *   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
 *   m_hat_t = m_t / (1 - beta1^t)
 *   v_hat_t = v_t / (1 - beta2^t)
 *   w_t = w_t - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)
 *
 * AdamW variant (when adamw=true):
 *   w_t = w_t - lr * (m_hat_t / (sqrt(v_hat_t) + epsilon) + weight_decay * w_t)
 *
 * Where:
 * - w_t: parameter at step t
 * - g_t: gradient at step t
 * - m_t: first moment (mean) estimate
 * - v_t: second moment (uncentered variance) estimate
 * - m_hat_t: bias-corrected first moment estimate
 * - v_hat_t: bias-corrected second moment estimate
 * - beta1, beta2: exponential decay rates for moment estimates
 * - epsilon: small constant for numerical stability
 *
 * References:
 * - Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014)
 * - Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2017) [AdamW]
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class Adam : public Optimizer<T> {
public:
    /**
     * @brief Constructs an Adam optimizer
     *
     * @param lr Learning rate (default: 0.001)
     * @param beta1 Coefficient for first moment estimate (default: 0.9)
     * @param beta2 Coefficient for second moment estimate (default: 0.999)
     * @param epsilon Small constant for numerical stability (default: 1e-8)
     * @param weight_decay Weight decay coefficient (default: 0.0)
     * @param adamw If true, use AdamW-style weight decay (default: false)
     */
    explicit Adam(T lr = T(0.001),
                  T beta1 = T(0.9),
                  T beta2 = T(0.999),
                  T epsilon = T(1e-8),
                  T weight_decay = T(0.0),
                  bool adamw = false)
        : lr_(lr),
          beta1_(beta1),
          beta2_(beta2),
          epsilon_(epsilon),
          weight_decay_(weight_decay),
          adamw_(adamw),
          step_count_(0) {
        if (lr <= 0) {
            throw std::invalid_argument("Learning rate must be positive");
        }
        if (beta1 < 0 || beta1 >= 1) {
            throw std::invalid_argument("Beta1 must be in [0, 1)");
        }
        if (beta2 < 0 || beta2 >= 1) {
            throw std::invalid_argument("Beta2 must be in [0, 1)");
        }
        if (epsilon <= 0) {
            throw std::invalid_argument("Epsilon must be positive");
        }
        if (weight_decay < 0) {
            throw std::invalid_argument("Weight decay must be non-negative");
        }
    }

    /**
     * @brief Performs a single optimization step
     *
     * Updates all parameters using their gradients and the Adam update rule.
     */
    void step() override {
        step_count_++;

        // Compute bias correction terms
        T bias_correction1 = T(1) - static_cast<T>(std::pow(static_cast<double>(beta1_),
                                                            static_cast<double>(step_count_)));
        T bias_correction2 = T(1) - static_cast<T>(std::pow(static_cast<double>(beta2_),
                                                            static_cast<double>(step_count_)));

        for (auto* param : this->params_) {
            if (!param->requiresGrad() || !param->hasGrad()) {
                continue;
            }

            // Get parameter data and gradient
            auto& data = param->data();
            const auto& grad = param->grad();

            // Get or create moment buffers for this parameter
            if (m_buffers_.find(param) == m_buffers_.end()) {
                m_buffers_[param] = Tensor<T>::zerosLike(data);
                v_buffers_[param] = Tensor<T>::zerosLike(data);
            }

            auto& m = m_buffers_[param];
            auto& v = v_buffers_[param];

            // Update moments and parameters element-wise
            for (size_t i = 0; i < data.size(); ++i) {
                T g = grad.data()[i];

                // Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                m.data()[i] = beta1_ * m.data()[i] + (T(1) - beta1_) * g;

                // Update biased second moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                v.data()[i] = beta2_ * v.data()[i] + (T(1) - beta2_) * g * g;

                // Compute bias-corrected moment estimates
                T m_hat = m.data()[i] / bias_correction1;
                T v_hat = v.data()[i] / bias_correction2;

                // Compute update: update = m_hat / (sqrt(v_hat) + epsilon)
                T update = m_hat / (std::sqrt(v_hat) + epsilon_);

                // Apply update with optional weight decay
                const T eps = T(1e-10);
                if (adamw_ && weight_decay_ > eps) {
                    // AdamW: apply weight decay directly to parameters
                    data.data()[i] -= lr_ * (update + weight_decay_ * data.data()[i]);
                } else if (weight_decay_ > eps) {
                    // Standard Adam with L2 regularization
                    update += weight_decay_ * data.data()[i];
                    data.data()[i] -= lr_ * update;
                } else {
                    // Standard Adam without weight decay
                    data.data()[i] -= lr_ * update;
                }
            }
        }
    }

    /**
     * @brief Returns the learning rate
     */
    T lr() const { return lr_; }

    /**
     * @brief Returns beta1 coefficient
     */
    T beta1() const { return beta1_; }

    /**
     * @brief Returns beta2 coefficient
     */
    T beta2() const { return beta2_; }

    /**
     * @brief Returns epsilon value
     */
    T epsilon() const { return epsilon_; }

    /**
     * @brief Returns weight decay coefficient
     */
    T weight_decay() const { return weight_decay_; }

    /**
     * @brief Returns whether AdamW variant is used
     */
    bool is_adamw() const { return adamw_; }

    /**
     * @brief Returns the current step count
     */
    size_t step_count() const { return step_count_; }

    /**
     * @brief Sets the learning rate
     */
    void set_lr(T lr) {
        if (lr <= 0) {
            throw std::invalid_argument("Learning rate must be positive");
        }
        lr_ = lr;
    }

private:
    T lr_;               ///< Learning rate
    T beta1_;            ///< Coefficient for first moment estimate
    T beta2_;            ///< Coefficient for second moment estimate
    T epsilon_;          ///< Small constant for numerical stability
    T weight_decay_;     ///< Weight decay coefficient
    bool adamw_;         ///< Whether to use AdamW variant
    size_t step_count_;  ///< Number of steps taken (for bias correction)

    /// First moment buffers (maps parameter pointer to its first moment)
    std::unordered_map<Variable<T>*, Tensor<T>> m_buffers_;

    /// Second moment buffers (maps parameter pointer to its second moment)
    std::unordered_map<Variable<T>*, Tensor<T>> v_buffers_;
};

}  // namespace optim
}  // namespace gradflow
