#pragma once

#include <gradflow/autograd/variable.hpp>
#include <memory>
#include <vector>

namespace gradflow::optim {

/**
 * @brief Base class for all optimizers
 *
 * Optimizer is responsible for updating model parameters based on their
 * gradients. This is the abstract base class that defines the common interface
 * for all optimization algorithms.
 *
 * Design principles:
 * - Template Pattern: Derived classes implement the step() method
 * - Non-owning: Optimizers hold pointers to parameters but don't own them
 * - Stateful: Optimizers maintain internal state (e.g., momentum buffers)
 *
 * Typical usage:
 * @code
 * // Create parameters
 * auto w = Variable<float>(Tensor<float>({2, 3}), true);
 * auto b = Variable<float>(Tensor<float>({2}), true);
 *
 * // Create optimizer
 * auto optimizer = std::make_unique<SGD<float>>(0.01);
 * optimizer->addParamGroup({&w, &b});
 *
 * // Training loop
 * for (int epoch = 0; epoch < 100; ++epoch) {
 *     // Forward pass
 *     auto loss = compute_loss(w, b);
 *
 *     // Backward pass
 *     optimizer->zeroGrad();
 *     loss.backward();
 *
 *     // Update parameters
 *     optimizer->step();
 * }
 * @endcode
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class Optimizer {
 public:
  /**
   * @brief Virtual destructor
   */
  virtual ~Optimizer() = default;

  // Optimizer instances should not be copied or moved
  Optimizer(const Optimizer&) = delete;
  Optimizer& operator=(const Optimizer&) = delete;
  Optimizer(Optimizer&&) = delete;
  Optimizer& operator=(Optimizer&&) = delete;

  /**
   * @brief Adds a parameter group to the optimizer
   *
   * Parameter groups allow different sets of parameters to have different
   * optimization settings (e.g., different learning rates).
   *
   * @param params Vector of pointers to variables to optimize
   */
  virtual void addParamGroup(const std::vector<Variable<T>*>& params) {
    params_.insert(params_.end(), params.begin(), params.end());
  }

  /**
   * @brief Performs a single optimization step
   *
   * This method updates all parameters based on their gradients.
   * Must be implemented by derived classes.
   */
  virtual void step() = 0;

  /**
   * @brief Zeros out the gradients of all parameters
   *
   * This should be called before each backward pass to prevent
   * gradient accumulation across iterations.
   */
  virtual void zeroGrad() {
    for (auto* param : params_) {
      if (param->requiresGrad()) {
        param->zeroGrad();
      }
    }
  }

  /**
   * @brief Returns the number of parameters being optimized
   */
  [[nodiscard]] size_t numParams() const { return params_.size(); }

  /**
   * @brief Returns the list of parameters being optimized
   */
  [[nodiscard]] const std::vector<Variable<T>*>& params() const {
    return params_;
  }

 protected:
  /**
   * @brief Protected constructor (only called by derived classes)
   */
  Optimizer() = default;

  /**
   * @brief List of parameters to optimize (non-owning pointers)
   */
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<Variable<T>*> params_;
};

}  // namespace gradflow::optim
