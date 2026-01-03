#include <gtest/gtest.h>

#include <cmath>
#include <gradflow/autograd/operation.hpp>
#include <gradflow/autograd/tensor.hpp>
#include <gradflow/autograd/variable.hpp>

namespace gradflow {
namespace test {

// ========================================
// Test Operations for Variable Testing
// ========================================

/**
 * @brief Simple addition operation for testing Variable
 *
 * f(x, y) = x + y
 * ∂f/∂x = 1, ∂f/∂y = 1
 */
template <typename T>
class AddOp : public Operation<T> {
 public:
  Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
    if (inputs.size() != 2) {
      throw std::invalid_argument("AddOp expects exactly 2 inputs");
    }

    const auto& x = inputs[0];
    const auto& y = inputs[1];

    if (x.shape() != y.shape()) {
      throw std::invalid_argument("Input shapes must match");
    }

    Tensor<T> output(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
      output.data()[i] = x.data()[i] + y.data()[i];
    }

    return output;
  }

  std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
    // ∂f/∂x = 1, ∂f/∂y = 1
    return {grad_output, grad_output};
  }

  std::string name() const override { return "AddOp"; }
};

/**
 * @brief Multiplication operation for testing Variable
 *
 * f(x, y) = x * y
 * ∂f/∂x = y, ∂f/∂y = x
 */
template <typename T>
class MulOp : public Operation<T> {
 public:
  Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
    if (inputs.size() != 2) {
      throw std::invalid_argument("MulOp expects exactly 2 inputs");
    }

    const auto& x = inputs[0];
    const auto& y = inputs[1];

    if (x.shape() != y.shape()) {
      throw std::invalid_argument("Input shapes must match");
    }

    // Save inputs for backward
    this->saveForBackward("x", x);
    this->saveForBackward("y", y);

    Tensor<T> output(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
      output.data()[i] = x.data()[i] * y.data()[i];
    }

    return output;
  }

  std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
    // ∂f/∂x = y, ∂f/∂y = x
    auto x = this->getSavedTensor("x");
    auto y = this->getSavedTensor("y");

    Tensor<T> grad_x(x.shape());
    Tensor<T> grad_y(y.shape());

    for (size_t i = 0; i < x.size(); ++i) {
      grad_x.data()[i] = grad_output.data()[i] * y.data()[i];
      grad_y.data()[i] = grad_output.data()[i] * x.data()[i];
    }

    return {grad_x, grad_y};
  }

  std::string name() const override { return "MulOp"; }
};

// ========================================
// Variable Construction Tests
// ========================================

TEST(VariableTest, Construction) {
  // Test default constructor
  // Note: Default Tensor() creates a scalar with size 1
  Variable<float> var1;
  EXPECT_EQ(var1.size(), 1);          // Scalar has size 1
  EXPECT_EQ(var1.shape().ndim(), 0);  // But 0 dimensions
  EXPECT_FALSE(var1.requiresGrad());
  EXPECT_TRUE(var1.isLeaf());

  // Test constructor with tensor
  Tensor<float> t(Shape({2, 3}));
  for (size_t i = 0; i < t.size(); ++i) {
    t.data()[i] = static_cast<float>(i);
  }

  Variable<float> var2(t, false);
  EXPECT_EQ(var2.shape(), Shape({2, 3}));
  EXPECT_EQ(var2.size(), 6);
  EXPECT_FALSE(var2.requiresGrad());
  EXPECT_TRUE(var2.isLeaf());
  EXPECT_FALSE(var2.hasGrad());

  // Test constructor with requires_grad=true
  Variable<float> var3(t, true);
  EXPECT_TRUE(var3.requiresGrad());
  EXPECT_TRUE(var3.isLeaf());
  EXPECT_FALSE(var3.hasGrad());

  // Test constructor with grad_fn
  auto op = std::make_shared<AddOp<float>>();
  Variable<float> var4(t, op, true);
  EXPECT_TRUE(var4.requiresGrad());
  EXPECT_FALSE(var4.isLeaf());
  EXPECT_EQ(var4.gradFn(), op);
}

TEST(VariableTest, DataAccess) {
  Tensor<float> t(Shape({2, 2}));
  t.data()[0] = 1.0f;
  t.data()[1] = 2.0f;
  t.data()[2] = 3.0f;
  t.data()[3] = 4.0f;

  Variable<float> var(t, false);

  // Test const data access
  const auto& data = var.data();
  EXPECT_FLOAT_EQ(data.data()[0], 1.0f);
  EXPECT_FLOAT_EQ(data.data()[1], 2.0f);
  EXPECT_FLOAT_EQ(data.data()[2], 3.0f);
  EXPECT_FLOAT_EQ(data.data()[3], 4.0f);

  // Test mutable data access
  var.data().data()[0] = 10.0f;
  EXPECT_FLOAT_EQ(var.data().data()[0], 10.0f);
}

TEST(VariableTest, RequiresGrad) {
  Tensor<float> t(Shape({2, 2}));
  Variable<float> var(t, false);

  EXPECT_FALSE(var.requiresGrad());

  var.setRequiresGrad(true);
  EXPECT_TRUE(var.requiresGrad());

  var.setRequiresGrad(false);
  EXPECT_FALSE(var.requiresGrad());
}

TEST(VariableTest, ZeroGrad) {
  Tensor<float> t(Shape({2, 2}));
  for (size_t i = 0; i < t.size(); ++i) {
    t.data()[i] = static_cast<float>(i);
  }

  Variable<float> var(t, true);

  // Initially no gradient
  EXPECT_FALSE(var.hasGrad());

  // Call zeroGrad - should create gradient tensor with zeros
  var.zeroGrad();
  EXPECT_TRUE(var.hasGrad());
  EXPECT_EQ(var.grad().shape(), var.shape());

  for (size_t i = 0; i < var.grad().size(); ++i) {
    EXPECT_FLOAT_EQ(var.grad().data()[i], 0.0f);
  }

  // Manually set gradient to non-zero
  for (size_t i = 0; i < var.grad().size(); ++i) {
    var.grad().data()[i] = static_cast<float>(i + 10);
  }

  // Call zeroGrad again - should zero out existing gradient
  var.zeroGrad();
  for (size_t i = 0; i < var.grad().size(); ++i) {
    EXPECT_FLOAT_EQ(var.grad().data()[i], 0.0f);
  }
}

// ========================================
// Gradient Accumulation Tests
// ========================================

TEST(VariableTest, GradAccumulation) {
  // Create leaf variable
  Tensor<float> t(Shape({2}));
  t.data()[0] = 1.0f;
  t.data()[1] = 2.0f;

  Variable<float> x(t, true);

  // Create first operation: y1 = x + x
  auto op1 = std::make_shared<AddOp<float>>();
  op1->setInputs({&x, &x});
  Tensor<float> y1_data = op1->forward({x.data(), x.data()});
  Variable<float> y1(y1_data, op1, true);

  // Backward from y1
  Tensor<float> grad1(Shape({2}));
  grad1.data()[0] = 1.0f;
  grad1.data()[1] = 1.0f;
  y1.backward(grad1);

  // Check gradient accumulation: x appears twice in computation
  // dy1/dx = 1 + 1 = 2 (from both inputs)
  EXPECT_TRUE(x.hasGrad());
  EXPECT_FLOAT_EQ(x.grad().data()[0], 2.0f);
  EXPECT_FLOAT_EQ(x.grad().data()[1], 2.0f);

  // Create second operation: y2 = x + x
  auto op2 = std::make_shared<AddOp<float>>();
  op2->setInputs({&x, &x});
  Tensor<float> y2_data = op2->forward({x.data(), x.data()});
  Variable<float> y2(y2_data, op2, true);

  // Backward from y2 (should accumulate with existing gradient)
  Tensor<float> grad2(Shape({2}));
  grad2.data()[0] = 1.0f;
  grad2.data()[1] = 1.0f;
  y2.backward(grad2);

  // Gradient should accumulate: 2 + 2 = 4
  EXPECT_FLOAT_EQ(x.grad().data()[0], 4.0f);
  EXPECT_FLOAT_EQ(x.grad().data()[1], 4.0f);
}

// ========================================
// Backward Pass Tests
// ========================================

TEST(VariableTest, BackwardSimple) {
  // Test simple backward: z = x + y
  Tensor<float> x_data(Shape({2}));
  x_data.data()[0] = 1.0f;
  x_data.data()[1] = 2.0f;

  Tensor<float> y_data(Shape({2}));
  y_data.data()[0] = 3.0f;
  y_data.data()[1] = 4.0f;

  Variable<float> x(x_data, true);
  Variable<float> y(y_data, true);

  // z = x + y
  auto add_op = std::make_shared<AddOp<float>>();
  add_op->setInputs({&x, &y});
  Tensor<float> z_data = add_op->forward({x.data(), y.data()});
  Variable<float> z(z_data, add_op, true);

  // Call backward with ones
  Tensor<float> grad_output(Shape({2}));
  grad_output.data()[0] = 1.0f;
  grad_output.data()[1] = 1.0f;
  z.backward(grad_output);

  // Check gradients
  EXPECT_TRUE(x.hasGrad());
  EXPECT_TRUE(y.hasGrad());

  // dz/dx = 1, dz/dy = 1
  EXPECT_FLOAT_EQ(x.grad().data()[0], 1.0f);
  EXPECT_FLOAT_EQ(x.grad().data()[1], 1.0f);
  EXPECT_FLOAT_EQ(y.grad().data()[0], 1.0f);
  EXPECT_FLOAT_EQ(y.grad().data()[1], 1.0f);
}

TEST(VariableTest, BackwardChain) {
  // Test chain rule: z = (x + y) * x
  // Let w = x + y, then z = w * x
  // dz/dx = dz/dw * dw/dx + dz/dx (direct)
  //       = x * 1 + w
  //       = x + (x + y)
  //       = 2x + y

  Tensor<float> x_data(Shape({1}));
  x_data.data()[0] = 2.0f;

  Tensor<float> y_data(Shape({1}));
  y_data.data()[0] = 3.0f;

  Variable<float> x(x_data, true);
  Variable<float> y(y_data, true);

  // w = x + y
  auto add_op = std::make_shared<AddOp<float>>();
  add_op->setInputs({&x, &y});
  Tensor<float> w_data = add_op->forward({x.data(), y.data()});
  Variable<float> w(w_data, add_op, true);

  // z = w * x
  auto mul_op = std::make_shared<MulOp<float>>();
  mul_op->setInputs({&w, &x});
  Tensor<float> z_data = mul_op->forward({w.data(), x.data()});
  Variable<float> z(z_data, mul_op, true);

  // Backward
  Tensor<float> grad_output(Shape({1}));
  grad_output.data()[0] = 1.0f;
  z.backward(grad_output);

  // Check gradients
  EXPECT_TRUE(x.hasGrad());
  EXPECT_TRUE(y.hasGrad());

  // dz/dx = 2x + y = 2*2 + 3 = 7
  EXPECT_NEAR(x.grad().data()[0], 7.0f, 1e-5);

  // dz/dy = x = 2
  EXPECT_NEAR(y.grad().data()[0], 2.0f, 1e-5);
}

TEST(VariableTest, BackwardWithoutRequiresGrad) {
  Tensor<float> t(Shape({2}));
  Variable<float> var(t, false);  // requires_grad=false

  // Calling backward should throw
  EXPECT_THROW(var.backward(), std::runtime_error);
}

TEST(VariableTest, BackwardOnLeafVariable) {
  // Test backward on leaf variable with requires_grad=true
  Tensor<float> t(Shape({2}));
  t.data()[0] = 1.0f;
  t.data()[1] = 2.0f;

  Variable<float> var(t, true);

  // Call backward - should initialize gradient to ones
  var.backward();

  EXPECT_TRUE(var.hasGrad());
  EXPECT_FLOAT_EQ(var.grad().data()[0], 1.0f);
  EXPECT_FLOAT_EQ(var.grad().data()[1], 1.0f);
}

TEST(VariableTest, GradAccessError) {
  Tensor<float> t(Shape({2}));
  Variable<float> var(t, false);

  // Accessing gradient before computation should throw
  EXPECT_THROW([[maybe_unused]] auto g = var.grad(), std::runtime_error);
}

TEST(VariableTest, BackwardShapeMismatch) {
  Tensor<float> t(Shape({2}));
  Variable<float> var(t, true);

  // Create gradient with wrong shape
  Tensor<float> wrong_grad(Shape({3}));

  // Calling backward with wrong shape should throw
  EXPECT_THROW(var.backward(wrong_grad), std::invalid_argument);
}

}  // namespace test
}  // namespace gradflow
