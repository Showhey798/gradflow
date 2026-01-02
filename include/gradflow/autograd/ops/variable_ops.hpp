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
 * @param x Input variable
 * @return Variable with ReLU applied
 */
template <typename T>
Variable<T> relu(const Variable<T>& x) {
    auto op = std::make_shared<ReLUOperation<T>>();
    op->setInputs({const_cast<Variable<T>*>(&x)});

    auto result_data = op->forward({x.data()});
    bool requires_grad = x.requiresGrad();

    return Variable<T>(result_data, op, requires_grad);
}

/**
 * @brief Applies Sigmoid activation to a Variable
 *
 * @tparam T Element type
 * @param x Input variable
 * @return Variable with Sigmoid applied
 */
template <typename T>
Variable<T> sigmoid(const Variable<T>& x) {
    auto op = std::make_shared<SigmoidOperation<T>>();
    op->setInputs({const_cast<Variable<T>*>(&x)});

    auto result_data = op->forward({x.data()});
    bool requires_grad = x.requiresGrad();

    return Variable<T>(result_data, op, requires_grad);
}

// ========================================
// Matrix Operations
// ========================================

/**
 * @brief Matrix multiplication for Variables
 *
 * @tparam T Element type
 * @param a First variable (M x K)
 * @param b Second variable (K x N)
 * @return Variable result (M x N)
 */
template <typename T>
Variable<T> matmul(const Variable<T>& a, const Variable<T>& b) {
    auto op = std::make_shared<MatMulOperation<T>>();
    op->setInputs({const_cast<Variable<T>*>(&a), const_cast<Variable<T>*>(&b)});

    auto result_data = op->forward({a.data(), b.data()});
    bool requires_grad = a.requiresGrad() || b.requiresGrad();

    return Variable<T>(result_data, op, requires_grad);
}

// ========================================
// Element-wise Operations
// ========================================

/**
 * @brief Addition operator for Variables
 *
 * @tparam T Element type
 * @param a First variable
 * @param b Second variable
 * @return Variable result (a + b)
 */
template <typename T>
Variable<T> operator+(const Variable<T>& a, const Variable<T>& b) {
    auto op = std::make_shared<AddOperation<T>>();
    op->setInputs({const_cast<Variable<T>*>(&a), const_cast<Variable<T>*>(&b)});

    auto result_data = op->forward({a.data(), b.data()});
    bool requires_grad = a.requiresGrad() || b.requiresGrad();

    return Variable<T>(result_data, op, requires_grad);
}

/**
 * @brief Multiplication operator for Variables
 *
 * @tparam T Element type
 * @param a First variable
 * @param b Second variable
 * @return Variable result (a * b)
 */
template <typename T>
Variable<T> operator*(const Variable<T>& a, const Variable<T>& b) {
    auto op = std::make_shared<MulOperation<T>>();
    op->setInputs({const_cast<Variable<T>*>(&a), const_cast<Variable<T>*>(&b)});

    auto result_data = op->forward({a.data(), b.data()});
    bool requires_grad = a.requiresGrad() || b.requiresGrad();

    return Variable<T>(result_data, op, requires_grad);
}

// ========================================
// Reduction Operations
// ========================================

/**
 * @brief Sum all elements of a Variable
 *
 * @tparam T Element type
 * @param x Input variable
 * @return Variable containing the scalar sum
 */
template <typename T>
Variable<T> sum(const Variable<T>& x) {
    auto op = std::make_shared<SumOperation<T>>();
    op->setInputs({const_cast<Variable<T>*>(&x)});

    auto result_data = op->forward({x.data()});
    bool requires_grad = x.requiresGrad();

    return Variable<T>(result_data, op, requires_grad);
}

// ========================================
// Loss Functions
// ========================================

/**
 * @brief Mean Squared Error loss for Variables
 *
 * @tparam T Element type
 * @param predicted Predicted values
 * @param target Target values
 * @return Variable representing the MSE loss (scalar)
 */
template <typename T>
Variable<T> mse_loss(const Variable<T>& predicted, const Variable<T>& target) {
    auto op = std::make_shared<MSELossOperation<T>>();
    op->setInputs({const_cast<Variable<T>*>(&predicted), const_cast<Variable<T>*>(&target)});

    auto result_data = op->forward({predicted.data(), target.data()});
    bool requires_grad = predicted.requiresGrad() || target.requiresGrad();

    return Variable<T>(result_data, op, requires_grad);
}

}  // namespace gradflow
