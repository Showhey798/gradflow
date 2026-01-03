#include <gtest/gtest.h>

#include <gradflow/autograd/operation.hpp>
#include <gradflow/autograd/tensor.hpp>

namespace gradflow {
namespace test {

// ========================================
// Test Operation Implementation
// ========================================

/**
 * @brief Simple test operation for testing the Operation base class
 *
 * This operation implements f(x, y) = x + y
 * ∂f/∂x = 1, ∂f/∂y = 1
 */
template <typename T>
class TestAddOperation : public Operation<T> {
 public:
  Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
    if (inputs.size() != 2) {
      throw std::invalid_argument("TestAddOperation expects exactly 2 inputs");
    }

    const auto& x = inputs[0];
    const auto& y = inputs[1];

    if (x.shape() != y.shape()) {
      throw std::invalid_argument("Input shapes must match");
    }

    // Save inputs for backward pass
    this->saveForBackward("x", x);
    this->saveForBackward("y", y);

    // Compute output
    Tensor<T> output(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
      output.data()[i] = x.data()[i] + y.data()[i];
    }

    return output;
  }

  std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
    // For addition: ∂f/∂x = 1, ∂f/∂y = 1
    // So both gradients equal grad_output
    return {grad_output, grad_output};
  }

  std::string name() const override { return "TestAddOperation"; }
};

// ========================================
// Operation Base Class Tests
// ========================================

TEST(OperationTest, SaveTensors) {
  // Create a test operation
  auto op = std::make_shared<TestAddOperation<float>>();

  // Create input tensors
  Tensor<float> x({2, 3});
  Tensor<float> y({2, 3});

  // Fill with test data
  for (size_t i = 0; i < x.size(); ++i) {
    x.data()[i] = static_cast<float>(i);
    y.data()[i] = static_cast<float>(i * 2);
  }

  // Execute forward pass (which saves tensors)
  auto output = op->forward({x, y});

  // Verify tensors were saved
  EXPECT_TRUE(op->hasSavedTensorForTest("x"));
  EXPECT_TRUE(op->hasSavedTensorForTest("y"));
  EXPECT_EQ(op->numSavedTensorsForTest(), 2);

  // Verify saved tensor values
  auto saved_x = op->getSavedTensorForTest("x");
  auto saved_y = op->getSavedTensorForTest("y");

  EXPECT_EQ(saved_x.shape(), x.shape());
  EXPECT_EQ(saved_y.shape(), y.shape());

  for (size_t i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(saved_x.data()[i], x.data()[i]);
    EXPECT_FLOAT_EQ(saved_y.data()[i], y.data()[i]);
  }
}

TEST(OperationTest, GetSavedTensors) {
  // Create a test operation
  auto op = std::make_shared<TestAddOperation<float>>();

  // Create input tensors with simple values
  Tensor<float> x({2});
  Tensor<float> y({2});

  x.data()[0] = 1.0f;
  x.data()[1] = 2.0f;
  y.data()[0] = 3.0f;
  y.data()[1] = 4.0f;

  // Execute forward pass
  auto output = op->forward({x, y});

  // Verify we can retrieve saved tensors
  EXPECT_TRUE(op->hasSavedTensorForTest("x"));
  EXPECT_TRUE(op->hasSavedTensorForTest("y"));
  EXPECT_EQ(op->numSavedTensorsForTest(), 2);

  // Get saved tensors
  auto saved_x = op->getSavedTensorForTest("x");
  auto saved_y = op->getSavedTensorForTest("y");

  // Verify saved values match original values
  EXPECT_EQ(saved_x.shape(), x.shape());
  EXPECT_FLOAT_EQ(saved_x.data()[0], 1.0f);
  EXPECT_FLOAT_EQ(saved_x.data()[1], 2.0f);

  EXPECT_EQ(saved_y.shape(), y.shape());
  EXPECT_FLOAT_EQ(saved_y.data()[0], 3.0f);
  EXPECT_FLOAT_EQ(saved_y.data()[1], 4.0f);
}

}  // namespace test
}  // namespace gradflow
