#pragma once

#include "operation.hpp"
#include "tensor.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

namespace gradflow {

/**
 * @brief Variable wraps a Tensor and enables automatic differentiation
 *
 * Variable is the core abstraction for automatic differentiation (autograd).
 * It wraps a Tensor and tracks the computational graph needed for backpropagation.
 *
 * Design principles:
 * - Wrapper Pattern: Variable wraps Tensor without duplicating functionality
 * - Graph Building: Maintains references to Operation nodes for backpropagation
 * - Lazy Evaluation: Gradients are computed only when backward() is called
 * - Gradient Accumulation: Multiple backward passes accumulate gradients
 *
 * Key members:
 * - data_: The underlying tensor containing the actual data
 * - grad_: The gradient tensor (same shape as data_)
 * - grad_fn_: Reference to the operation that created this variable
 * - requires_grad_: Flag indicating if gradients should be computed
 *
 * Usage example:
 * @code
 * // Create variables
 * auto x = Variable<float>(Tensor<float>({2, 3}), true);  // requires_grad=true
 * auto y = Variable<float>(Tensor<float>({2, 3}), true);
 *
 * // Perform operations (builds computational graph)
 * auto z = x + y;  // z.grad_fn_ points to AddOperation
 *
 * // Compute gradients
 * z.backward();  // Propagates gradients back to x and y
 *
 * // Access gradients
 * auto x_grad = x.grad();  // Gradient of loss with respect to x
 * @endcode
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class Variable {
public:
    /**
     * @brief Default constructor - creates an empty variable
     */
    Variable() : data_(), grad_(), grad_fn_(nullptr), requires_grad_(false) {}

    /**
     * @brief Constructs a Variable from a Tensor
     *
     * @param data The underlying tensor
     * @param requires_grad Whether to compute gradients for this variable (default: false)
     */
    explicit Variable(const Tensor<T>& data, bool requires_grad = false)
        : data_(data),
          grad_(),
          grad_fn_(nullptr),
          requires_grad_(requires_grad) {}

    /**
     * @brief Constructs a Variable with specified gradient function
     *
     * This constructor is typically used internally when creating variables
     * as outputs of operations.
     *
     * @param data The underlying tensor
     * @param grad_fn The operation that created this variable
     * @param requires_grad Whether to compute gradients for this variable
     */
    Variable(const Tensor<T>& data,
             std::shared_ptr<Operation<T>> grad_fn,
             bool requires_grad = true)
        : data_(data),
          grad_(),
          grad_fn_(std::move(grad_fn)),
          requires_grad_(requires_grad) {}

    // Copy constructor and assignment operator
    Variable(const Variable&) = default;
    Variable& operator=(const Variable&) = default;

    // Move constructor and assignment operator
    Variable(Variable&&) noexcept = default;
    Variable& operator=(Variable&&) noexcept = default;

    /**
     * @brief Destructor
     */
    ~Variable() = default;

    /**
     * @brief Returns the underlying data tensor
     *
     * @return Reference to the data tensor
     */
    [[nodiscard]] const Tensor<T>& data() const { return data_; }

    /**
     * @brief Returns the underlying data tensor (mutable)
     *
     * @return Reference to the data tensor
     */
    [[nodiscard]] Tensor<T>& data() { return data_; }

    /**
     * @brief Returns the gradient tensor
     *
     * The gradient tensor has the same shape as the data tensor and contains
     * the gradient of the loss with respect to this variable.
     *
     * @return Reference to the gradient tensor
     * @throws std::runtime_error if gradient has not been computed
     */
    [[nodiscard]] const Tensor<T>& grad() const {
        if (!hasGrad()) {
            throw std::runtime_error("Gradient has not been computed. Call backward() first.");
        }
        return grad_;
    }

    /**
     * @brief Returns the gradient tensor (mutable)
     *
     * @return Reference to the gradient tensor
     * @throws std::runtime_error if gradient has not been computed
     */
    [[nodiscard]] Tensor<T>& grad() {
        if (!hasGrad()) {
            throw std::runtime_error("Gradient has not been computed. Call backward() first.");
        }
        return grad_;
    }

    /**
     * @brief Checks if gradient has been computed
     *
     * @return True if gradient tensor exists and has been computed
     */
    [[nodiscard]] bool hasGrad() const {
        // Check if grad_ has been initialized with non-zero shape
        return grad_.shape().ndim() > 0 && grad_.size() > 0;
    }

    /**
     * @brief Returns whether this variable requires gradient computation
     *
     * @return True if gradients should be computed for this variable
     */
    [[nodiscard]] bool requiresGrad() const { return requires_grad_; }

    /**
     * @brief Sets whether this variable requires gradient computation
     *
     * @param requires_grad New value for requires_grad flag
     */
    void setRequiresGrad(bool requires_grad) { requires_grad_ = requires_grad; }

    /**
     * @brief Returns the gradient function (operation that created this variable)
     *
     * @return Shared pointer to the gradient function (nullptr if this is a leaf variable)
     */
    [[nodiscard]] const std::shared_ptr<Operation<T>>& gradFn() const { return grad_fn_; }

    /**
     * @brief Sets the gradient function
     *
     * This is typically called when this variable is created as the output of an operation.
     *
     * @param grad_fn The operation that created this variable
     */
    void setGradFn(std::shared_ptr<Operation<T>> grad_fn) { grad_fn_ = std::move(grad_fn); }

    /**
     * @brief Checks if this is a leaf variable
     *
     * A leaf variable is one that was created directly by the user (not as a result
     * of an operation). Leaf variables have grad_fn_ == nullptr.
     *
     * @return True if this is a leaf variable
     */
    [[nodiscard]] bool isLeaf() const { return grad_fn_ == nullptr; }

    /**
     * @brief Zeros the gradient tensor
     *
     * Sets all elements of the gradient tensor to zero. This is typically called
     * before each backward pass in training loops.
     */
    void zeroGrad() {
        if (hasGrad()) {
            // Zero out existing gradient
            for (size_t i = 0; i < grad_.size(); ++i) {
                grad_.data()[i] = T(0);
            }
        } else {
            // Create zero gradient with same shape as data
            grad_ = Tensor<T>(data_.shape());
            for (size_t i = 0; i < grad_.size(); ++i) {
                grad_.data()[i] = T(0);
            }
        }
    }

    /**
     * @brief Computes gradients using backpropagation
     *
     * This method implements automatic differentiation by traversing the computational
     * graph in reverse topological order and applying the chain rule.
     *
     * Algorithm:
     * 1. Initialize gradient of this variable to 1 (∂L/∂L = 1)
     * 2. Traverse the graph backwards using DFS/BFS
     * 3. For each operation, compute gradients w.r.t. inputs using chain rule
     * 4. Accumulate gradients at each variable
     *
     * The gradient with respect to this variable is set to a tensor of ones
     * (representing dL/dL = 1, where L is this variable treated as the loss).
     *
     * @param retain_graph If true, keep the computational graph for multiple backward passes
     *                     (default: false)
     * @throws std::runtime_error if requires_grad is false
     */
    void backward(bool retain_graph = false) {
        if (!requires_grad_) {
            throw std::runtime_error(
                "Cannot call backward() on a variable that doesn't require gradients. "
                "Set requires_grad=true when creating the variable.");
        }

        // Initialize gradient to ones (∂L/∂L = 1)
        if (!hasGrad()) {
            grad_ = Tensor<T>(data_.shape());
        }
        for (size_t i = 0; i < grad_.size(); ++i) {
            grad_.data()[i] = T(1);
        }

        // Perform backward pass through the computational graph
        backwardImpl(grad_, retain_graph);
    }

    /**
     * @brief Computes gradients with a specified gradient tensor
     *
     * This is useful when this variable is not the final loss (scalar) but an
     * intermediate tensor, and you want to propagate a specific gradient.
     *
     * @param grad The gradient tensor to propagate (must have same shape as data)
     * @param retain_graph If true, keep the computational graph for multiple backward passes
     * @throws std::invalid_argument if grad shape doesn't match data shape
     * @throws std::runtime_error if requires_grad is false
     */
    void backward(const Tensor<T>& grad, bool retain_graph = false) {
        if (!requires_grad_) {
            throw std::runtime_error(
                "Cannot call backward() on a variable that doesn't require "
                "gradients.");
        }

        if (grad.shape() != data_.shape()) {
            throw std::invalid_argument("Gradient shape must match data shape");
        }

        // Accumulate the provided gradient
        if (!hasGrad()) {
            // Create a new tensor for gradient (deep copy to avoid sharing storage)
            grad_ = Tensor<T>(grad.shape());
            for (size_t i = 0; i < grad.size(); ++i) {
                grad_.data()[i] = grad.data()[i];
            }
        } else {
            // Accumulate: grad_ += grad
            for (size_t i = 0; i < grad_.size(); ++i) {
                grad_.data()[i] += grad.data()[i];
            }
        }

        // Propagate gradient backwards
        backwardImpl(grad, retain_graph);
    }

    /**
     * @brief Returns the shape of the underlying tensor
     *
     * @return Shape of the data tensor
     */
    [[nodiscard]] const Shape& shape() const { return data_.shape(); }

    /**
     * @brief Returns the number of elements in the tensor
     *
     * @return Total number of elements
     */
    [[nodiscard]] size_t size() const { return data_.size(); }

private:
    /**
     * @brief Internal implementation of backward pass
     *
     * This method recursively traverses the computational graph and computes gradients.
     *
     * @param grad_output Gradient of loss with respect to this variable
     * @param retain_graph Whether to retain the graph for future backward passes
     */
    void backwardImpl(const Tensor<T>& grad_output, bool retain_graph) {
        // If this is a leaf variable (no grad_fn), we're done
        if (isLeaf()) {
            return;
        }

        // Compute gradients with respect to inputs
        auto input_grads = grad_fn_->backward(grad_output);

        // Get input variables from the operation
        const auto& inputs = grad_fn_->inputs();

        // Propagate gradients to input variables
        if (inputs.size() != input_grads.size()) {
            throw std::runtime_error("Mismatch between number of inputs and computed gradients");
        }

        for (size_t i = 0; i < inputs.size(); ++i) {
            auto* input_var = inputs[i];
            if (input_var != nullptr && input_var->requiresGrad()) {
                // Accumulate gradient
                if (!input_var->hasGrad()) {
                    // Create a new tensor for gradient (deep copy to avoid sharing storage)
                    input_var->grad_ = Tensor<T>(input_grads[i].shape());
                    for (size_t j = 0; j < input_grads[i].size(); ++j) {
                        input_var->grad_.data()[j] = input_grads[i].data()[j];
                    }
                } else {
                    // Gradient accumulation: grad += new_grad
                    for (size_t j = 0; j < input_grads[i].size(); ++j) {
                        input_var->grad_.data()[j] += input_grads[i].data()[j];
                    }
                }

                // Recursively call backward on input variables
                if (!input_var->isLeaf()) {
                    input_var->backwardImpl(input_grads[i], retain_graph);
                }
            }
        }

        // If not retaining graph, clear the operation's saved tensors to free memory
        if (!retain_graph) {
            grad_fn_->clearSavedTensorsForTest();
        }
    }

    /**
     * @brief The underlying data tensor
     */
    Tensor<T> data_;

    /**
     * @brief The gradient tensor (∂Loss/∂data)
     *
     * Has the same shape as data_. Accumulated across multiple backward passes.
     */
    Tensor<T> grad_;

    /**
     * @brief The operation that created this variable (gradient function)
     *
     * For leaf variables (created directly by user), this is nullptr.
     * For intermediate variables (created by operations), this points to the operation.
     */
    std::shared_ptr<Operation<T>> grad_fn_;

    /**
     * @brief Flag indicating whether to compute gradients for this variable
     *
     * If false, this variable and its dependencies will be excluded from the
     * computational graph.
     */
    bool requires_grad_;
};

}  // namespace gradflow
