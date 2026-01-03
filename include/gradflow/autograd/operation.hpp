#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.hpp"

namespace gradflow {

// Forward declaration
template <typename T>
class Variable;

/**
 * @brief Abstract base class for all operations in the computational graph
 *
 * Operation represents a computation node in the automatic differentiation
 * graph. Each operation implements forward and backward passes for gradient
 * computation.
 *
 * Design principles:
 * - Single Responsibility: Each operation class handles one specific
 * computation
 * - Template Method Pattern: Base class defines the structure, derived classes
 * implement specifics
 * - RAII: Automatic cleanup of saved tensors
 *
 * Usage:
 * 1. Derive from Operation<T>
 * 2. Implement forward() to compute the output from inputs
 * 3. Implement backward() to compute gradients with respect to inputs
 * 4. Use save_for_backward() to store intermediate values needed for backward
 * pass
 *
 * Example:
 * @code
 * template <typename T>
 * class AddOperation : public Operation<T> {
 * public:
 *     Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
 *         return inputs[0] + inputs[1];
 *     }
 *
 *     std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
 *         return {grad_output, grad_output};
 *     }
 * };
 * @endcode
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class Operation : public std::enable_shared_from_this<Operation<T>> {
 public:
  virtual ~Operation() = default;

  // Delete copy operations (operations should not be copied)
  Operation(const Operation&) = delete;
  Operation& operator=(const Operation&) = delete;

  // Delete move operations (operations should not be moved)
  Operation(Operation&&) = delete;
  Operation& operator=(Operation&&) = delete;

  /**
   * @brief Forward pass: compute output from inputs
   *
   * This method performs the forward computation of the operation.
   * It should:
   * 1. Validate inputs
   * 2. Perform the computation
   * 3. Save any intermediate values needed for backward pass using
   * saveForBackward()
   *
   * @param inputs Vector of input tensors
   * @return Output tensor
   * @throws std::invalid_argument if inputs are invalid
   */
  virtual Tensor<T> forward(const std::vector<Tensor<T>>& inputs) = 0;

  /**
   * @brief Backward pass: compute gradients with respect to inputs
   *
   * This method computes the gradients of the loss with respect to the inputs.
   * Given the gradient of the loss with respect to the output (grad_output),
   * it should return the gradients with respect to each input.
   *
   * Chain rule:
   * If y = f(x1, x2, ...), then:
   * ∂L/∂xi = ∂L/∂y * ∂y/∂xi
   *
   * where:
   * - grad_output = ∂L/∂y
   * - return[i] = ∂L/∂xi
   *
   * @param grad_output Gradient of loss with respect to output
   * @return Vector of gradients with respect to each input
   */
  virtual std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) = 0;

  /**
   * @brief Returns the input variables of this operation
   *
   * These are the variables that were used as inputs during the forward pass.
   * They are stored to maintain the computational graph structure.
   *
   * @return Vector of input variables (empty if not set)
   */
  const std::vector<Variable<T>*>& inputs() const { return inputs_; }

  /**
   * @brief Sets the input variables for this operation
   *
   * This is typically called automatically when the operation is executed
   * as part of building the computational graph.
   *
   * @param inputs Vector of input variables
   */
  void setInputs(const std::vector<Variable<T>*>& inputs) { inputs_ = inputs; }

  /**
   * @brief Returns the name of this operation (for debugging)
   *
   * Default implementation returns the demangled type name.
   * Can be overridden for custom names.
   *
   * @return Operation name
   */
  [[nodiscard]] virtual std::string name() const { return "Operation"; }

  /**
   * @brief Check if a tensor with given name was saved (public for testing)
   *
   * @param name Identifier to check
   * @return True if tensor exists
   */
  [[nodiscard]] bool hasSavedTensorForTest(const std::string& name) const {
    return saved_tensors_.find(name) != saved_tensors_.end();
  }

  /**
   * @brief Returns the number of saved tensors (public for testing)
   *
   * Useful for testing and debugging.
   *
   * @return Number of saved tensors
   */
  [[nodiscard]] size_t numSavedTensorsForTest() const {
    return saved_tensors_.size();
  }

  /**
   * @brief Retrieve a tensor saved during forward pass (public for testing)
   *
   * @param name Identifier of the saved tensor
   * @return The saved tensor
   * @throws std::out_of_range if name not found
   */
  [[nodiscard]] const Tensor<T>& getSavedTensorForTest(
      const std::string& name) const {
    return saved_tensors_.at(name);
  }

  /**
   * @brief Clear all saved tensors (public for testing)
   *
   * This can be called after backward pass to free memory.
   * Note: Tensors are automatically cleaned up when the operation is destroyed.
   */
  void clearSavedTensorsForTest() { saved_tensors_.clear(); }

 protected:
  Operation() = default;

  /**
   * @brief Save a tensor for use in backward pass
   *
   * During the forward pass, operations often need to save intermediate values
   * (inputs, outputs, or computed values) that will be needed during the
   * backward pass.
   *
   * Example:
   * @code
   * // In forward():
   * auto output = compute_output(inputs);
   * saveForBackward("input", inputs[0]);
   * saveForBackward("output", output);
   * return output;
   *
   * // In backward():
   * auto input = getSavedTensor("input");
   * auto output = getSavedTensor("output");
   * // compute gradients using saved values
   * @endcode
   *
   * @param name Identifier for the saved tensor
   * @param tensor Tensor to save
   */
  void saveForBackward(const std::string& name, const Tensor<T>& tensor) {
    saved_tensors_[name] = tensor;
  }

  /**
   * @brief Retrieve a tensor saved during forward pass
   *
   * @param name Identifier of the saved tensor
   * @return The saved tensor
   * @throws std::out_of_range if name not found
   */
  [[nodiscard]] const Tensor<T>& getSavedTensor(const std::string& name) const {
    return saved_tensors_.at(name);
  }

  /**
   * @brief Check if a tensor with given name was saved
   *
   * @param name Identifier to check
   * @return True if tensor exists
   */
  [[nodiscard]] bool hasSavedTensor(const std::string& name) const {
    return saved_tensors_.find(name) != saved_tensors_.end();
  }

  /**
   * @brief Clear all saved tensors
   *
   * This can be called after backward pass to free memory.
   * Note: Tensors are automatically cleaned up when the operation is destroyed.
   */
  void clearSavedTensors() { saved_tensors_.clear(); }

  /**
   * @brief Returns the number of saved tensors
   *
   * Useful for testing and debugging.
   *
   * @return Number of saved tensors
   */
  [[nodiscard]] size_t numSavedTensors() const { return saved_tensors_.size(); }

 private:
  /**
   * @brief Input variables in the computational graph
   *
   * These are raw pointers (not owned) to the variables that were inputs
   * to this operation. They are used to traverse the graph during
   * backpropagation.
   */
  std::vector<Variable<T>*> inputs_;

  /**
   * @brief Tensors saved during forward pass for use in backward pass
   *
   * Key: identifier string
   * Value: saved tensor
   *
   * Memory management:
   * - Tensors use shared_ptr internally, so copies are cheap
   * - Saved tensors are automatically cleaned up when operation is destroyed
   * - Can be manually cleared with clear_saved_tensors()
   */
  std::unordered_map<std::string, Tensor<T>> saved_tensors_;
};

}  // namespace gradflow
