#pragma once

#include "../variable.hpp"
#include "add.hpp"
#include "loss.hpp"
#include "matmul_op.hpp"
#include "mul.hpp"
#include "reduction.hpp"
#include "relu.hpp"
#include "sigmoid.hpp"

#include <memory>

namespace gradflow {

/**
 * @brief Variable helper functions for automatic differentiation
 *
 * This file provides convenient helper functions to apply operations
 * to Variable objects while maintaining the computational graph.
 */

// ========================================
// Activation Functions
// ========================================

/**
 * @brief Applies ReLU activation to a Variable
 *
 * @tparam T Element type
 * @param x Input variable (non-const reference to build computational graph)
 * @return Variable with ReLU applied
 */
template <typename T>
Variable<T> relu(Variable<T>& x) {
    auto op = std::make_shared<ReLUOperation<T>>();
    op->setInputs({&x});

    auto result_data = op->forward({x.data()});
    bool requires_grad = x.requiresGrad();

    return Variable<T>(std::move(result_data), std::move(op), requires_grad);
}

/**
 * @brief Applies Sigmoid activation to a Variable
 *
 * @tparam T Element type
 * @param x Input variable (non-const reference to build computational graph)
 * @return Variable with Sigmoid applied
 */
template <typename T>
Variable<T> sigmoid(Variable<T>& x) {
    auto op = std::make_shared<SigmoidOperation<T>>();
    op->setInputs({&x});

    auto result_data = op->forward({x.data()});
    bool requires_grad = x.requiresGrad();

    return Variable<T>(std::move(result_data), std::move(op), requires_grad);
}

// ========================================
// Matrix Operations
// ========================================

/**
 * @brief Matrix multiplication for Variables
 *
 * @tparam T Element type
 * @param a First variable (M x K) (non-const reference to build computational graph)
 * @param b Second variable (K x N) (non-const reference to build computational graph)
 * @return Variable result (M x N)
 */
template <typename T>
Variable<T> matmul(Variable<T>& a, Variable<T>& b) {
    auto op = std::make_shared<MatMulOperation<T>>();
    op->setInputs({&a, &b});

    auto result_data = op->forward({a.data(), b.data()});
    bool requires_grad = a.requiresGrad() || b.requiresGrad();

    return Variable<T>(std::move(result_data), std::move(op), requires_grad);
}

// ========================================
// Element-wise Operations
// ========================================

/**
 * @brief Addition operator for Variables
 *
 * @tparam T Element type
 * @param a First variable (non-const reference to build computational graph)
 * @param b Second variable (non-const reference to build computational graph)
 * @return Variable result (a + b)
 */
template <typename T>
Variable<T> operator+(Variable<T>& a, Variable<T>& b) {
    auto op = std::make_shared<AddOperation<T>>();
    op->setInputs({&a, &b});

    auto result_data = op->forward({a.data(), b.data()});
    bool requires_grad = a.requiresGrad() || b.requiresGrad();

    return Variable<T>(std::move(result_data), std::move(op), requires_grad);
}

/**
 * @brief Multiplication operator for Variables
 *
 * @tparam T Element type
 * @param a First variable (non-const reference to build computational graph)
 * @param b Second variable (non-const reference to build computational graph)
 * @return Variable result (a * b)
 */
template <typename T>
Variable<T> operator*(Variable<T>& a, Variable<T>& b) {
    auto op = std::make_shared<MulOperation<T>>();
    op->setInputs({&a, &b});

    auto result_data = op->forward({a.data(), b.data()});
    bool requires_grad = a.requiresGrad() || b.requiresGrad();

    return Variable<T>(std::move(result_data), std::move(op), requires_grad);
}

// ========================================
// Reduction Operations
// ========================================

/**
 * @brief Sum all elements of a Variable
 *
 * @tparam T Element type
 * @param x Input variable (non-const reference to build computational graph)
 * @return Variable containing the scalar sum
 */
template <typename T>
Variable<T> sum(Variable<T>& x) {
    auto op = std::make_shared<SumOperation<T>>();
    op->setInputs({&x});

    auto result_data = op->forward({x.data()});
    bool requires_grad = x.requiresGrad();

    return Variable<T>(std::move(result_data), std::move(op), requires_grad);
}

// ========================================
// Loss Functions
// ========================================

/**
 * @brief Mean Squared Error loss for Variables
 *
 * @tparam T Element type
 * @param predicted Predicted values (non-const reference to build computational graph)
 * @param target Target values (non-const reference to build computational graph)
 * @return Variable representing the MSE loss (scalar)
 */
template <typename T>
Variable<T> mse_loss(Variable<T>& predicted, Variable<T>& target) {
    auto op = std::make_shared<MSELossOperation<T>>();
    op->setInputs({&predicted, &target});

    auto result_data = op->forward({predicted.data(), target.data()});
    bool requires_grad = predicted.requiresGrad() || target.requiresGrad();

    return Variable<T>(std::move(result_data), std::move(op), requires_grad);
}

}  // namespace gradflow
