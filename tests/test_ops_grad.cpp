#include "gradflow/autograd/ops/add.hpp"
#include "gradflow/autograd/ops/div.hpp"
#include "gradflow/autograd/ops/exp.hpp"
#include "gradflow/autograd/ops/log.hpp"
#include "gradflow/autograd/ops/matmul_op.hpp"
#include "gradflow/autograd/ops/mul.hpp"
#include "gradflow/autograd/ops/op_utils.hpp"
#include "gradflow/autograd/ops/pow.hpp"
#include "gradflow/autograd/ops/sqrt.hpp"
#include "gradflow/autograd/ops/sub.hpp"

#include <cmath>

#include <gtest/gtest.h>

using namespace gradflow;

class OpsGradTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper function to check if two floats are approximately equal
    bool approx_equal(float a, float b, float epsilon = 1e-5F) { return std::abs(a - b) < epsilon; }
};

// ========================================
// AddOperation Tests
// ========================================

TEST_F(OpsGradTest, AddForward) {
    auto x = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto y = Tensor<float>({4.0F, 5.0F, 6.0F});

    auto op = std::make_shared<AddOperation<float>>();
    auto result = op->forward({x, y});

    EXPECT_EQ(result.shape(), Shape({3}));
    EXPECT_FLOAT_EQ(result[{0}], 5.0F);
    EXPECT_FLOAT_EQ(result[{1}], 7.0F);
    EXPECT_FLOAT_EQ(result[{2}], 9.0F);
}

TEST_F(OpsGradTest, AddBackward) {
    auto x = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto y = Tensor<float>({4.0F, 5.0F, 6.0F});

    auto op = std::make_shared<AddOperation<float>>();
    auto result = op->forward({x, y});

    auto grad_output = Tensor<float>({1.0F, 1.0F, 1.0F});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 2UL);

    // For addition, gradients are passed through
    EXPECT_EQ(grads[0].shape(), x.shape());
    EXPECT_FLOAT_EQ(grads[0][{0}], 1.0F);
    EXPECT_FLOAT_EQ(grads[0][{1}], 1.0F);
    EXPECT_FLOAT_EQ(grads[0][{2}], 1.0F);

    EXPECT_EQ(grads[1].shape(), y.shape());
    EXPECT_FLOAT_EQ(grads[1][{0}], 1.0F);
    EXPECT_FLOAT_EQ(grads[1][{1}], 1.0F);
    EXPECT_FLOAT_EQ(grads[1][{2}], 1.0F);
}

TEST_F(OpsGradTest, AddBroadcastBackward) {
    auto x = Tensor<float>({{1.0F, 2.0F}, {3.0F, 4.0F}});  // [2, 2]
    auto y = Tensor<float>({10.0F, 20.0F});                // [2]

    auto op = std::make_shared<AddOperation<float>>();
    auto result = op->forward({x, y});  // Result: [2, 2]

    EXPECT_EQ(result.shape(), Shape({2, 2}));

    auto grad_output = Tensor<float>({{1.0F, 1.0F}, {1.0F, 1.0F}});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 2UL);

    // grad_x should be [2, 2]
    EXPECT_EQ(grads[0].shape(), x.shape());
    EXPECT_FLOAT_EQ((grads[0][{0, 0}]), 1.0F);
    EXPECT_FLOAT_EQ((grads[0][{0, 1}]), 1.0F);
    EXPECT_FLOAT_EQ((grads[0][{1, 0}]), 1.0F);
    EXPECT_FLOAT_EQ((grads[0][{1, 1}]), 1.0F);

    // grad_y should be [2], summed over axis 0
    EXPECT_EQ(grads[1].shape(), y.shape());
    EXPECT_FLOAT_EQ(grads[1][{0}], 2.0F);  // Sum of [1.0, 1.0]
    EXPECT_FLOAT_EQ(grads[1][{1}], 2.0F);  // Sum of [1.0, 1.0]
}

TEST_F(OpsGradTest, AddNumericalGradient) {
    auto x = Tensor<float>({2.0F, 3.0F});
    auto y = Tensor<float>({4.0F, 5.0F});

    auto op = std::make_shared<AddOperation<float>>();

    // Perform forward and backward
    auto output = op->forward({x, y});

    // Check numerical gradient
    bool passed = ops::test::checkNumericalGradient(*op, {x, y});
    EXPECT_TRUE(passed);
}

// ========================================
// SubOperation Tests
// ========================================

TEST_F(OpsGradTest, SubForward) {
    auto x = Tensor<float>({5.0F, 7.0F, 9.0F});
    auto y = Tensor<float>({1.0F, 2.0F, 3.0F});

    auto op = std::make_shared<SubOperation<float>>();
    auto result = op->forward({x, y});

    EXPECT_EQ(result.shape(), Shape({3}));
    EXPECT_FLOAT_EQ(result[{0}], 4.0F);
    EXPECT_FLOAT_EQ(result[{1}], 5.0F);
    EXPECT_FLOAT_EQ(result[{2}], 6.0F);
}

TEST_F(OpsGradTest, SubBackward) {
    auto x = Tensor<float>({5.0F, 7.0F});
    auto y = Tensor<float>({1.0F, 2.0F});

    auto op = std::make_shared<SubOperation<float>>();
    auto result = op->forward({x, y});

    auto grad_output = Tensor<float>({1.0F, 1.0F});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 2UL);

    // grad_x = grad_output
    EXPECT_EQ(grads[0].shape(), x.shape());
    EXPECT_FLOAT_EQ(grads[0][{0}], 1.0F);
    EXPECT_FLOAT_EQ(grads[0][{1}], 1.0F);

    // grad_y = -grad_output
    EXPECT_EQ(grads[1].shape(), y.shape());
    EXPECT_FLOAT_EQ(grads[1][{0}], -1.0F);
    EXPECT_FLOAT_EQ(grads[1][{1}], -1.0F);
}

TEST_F(OpsGradTest, SubNumericalGradient) {
    auto x = Tensor<float>({5.0F, 7.0F});
    auto y = Tensor<float>({1.0F, 2.0F});

    auto op = std::make_shared<SubOperation<float>>();
    bool passed = ops::test::checkNumericalGradient(*op, {x, y});
    EXPECT_TRUE(passed);
}

// ========================================
// MulOperation Tests
// ========================================

TEST_F(OpsGradTest, MulForward) {
    auto x = Tensor<float>({2.0F, 3.0F, 4.0F});
    auto y = Tensor<float>({5.0F, 6.0F, 7.0F});

    auto op = std::make_shared<MulOperation<float>>();
    auto result = op->forward({x, y});

    EXPECT_EQ(result.shape(), Shape({3}));
    EXPECT_FLOAT_EQ(result[{0}], 10.0F);
    EXPECT_FLOAT_EQ(result[{1}], 18.0F);
    EXPECT_FLOAT_EQ(result[{2}], 28.0F);
}

TEST_F(OpsGradTest, MulBackward) {
    auto x = Tensor<float>({2.0F, 3.0F});
    auto y = Tensor<float>({4.0F, 5.0F});

    auto op = std::make_shared<MulOperation<float>>();
    auto result = op->forward({x, y});

    auto grad_output = Tensor<float>({1.0F, 1.0F});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 2UL);

    // grad_x = grad_output * y
    EXPECT_EQ(grads[0].shape(), x.shape());
    EXPECT_FLOAT_EQ(grads[0][{0}], 4.0F);
    EXPECT_FLOAT_EQ(grads[0][{1}], 5.0F);

    // grad_y = grad_output * x
    EXPECT_EQ(grads[1].shape(), y.shape());
    EXPECT_FLOAT_EQ(grads[1][{0}], 2.0F);
    EXPECT_FLOAT_EQ(grads[1][{1}], 3.0F);
}

TEST_F(OpsGradTest, MulNumericalGradient) {
    auto x = Tensor<float>({2.0F, 3.0F});
    auto y = Tensor<float>({4.0F, 5.0F});

    auto op = std::make_shared<MulOperation<float>>();
    bool passed = ops::test::checkNumericalGradient(*op, {x, y});
    EXPECT_TRUE(passed);
}

// ========================================
// DivOperation Tests
// ========================================

TEST_F(OpsGradTest, DivForward) {
    auto x = Tensor<float>({10.0F, 20.0F, 30.0F});
    auto y = Tensor<float>({2.0F, 4.0F, 5.0F});

    auto op = std::make_shared<DivOperation<float>>();
    auto result = op->forward({x, y});

    EXPECT_EQ(result.shape(), Shape({3}));
    EXPECT_FLOAT_EQ(result[{0}], 5.0F);
    EXPECT_FLOAT_EQ(result[{1}], 5.0F);
    EXPECT_FLOAT_EQ(result[{2}], 6.0F);
}

TEST_F(OpsGradTest, DivNumericalGradient) {
    auto x = Tensor<float>({10.0F, 20.0F});
    auto y = Tensor<float>({2.0F, 4.0F});

    auto op = std::make_shared<DivOperation<float>>();
    bool passed = ops::test::checkNumericalGradient(*op, {x, y});
    EXPECT_TRUE(passed);
}

// ========================================
// MatMulOperation Tests
// ========================================

TEST_F(OpsGradTest, MatMulForward) {
    auto x = Tensor<float>({{1.0F, 2.0F}, {3.0F, 4.0F}});  // [2, 2]
    auto y = Tensor<float>({{5.0F, 6.0F}, {7.0F, 8.0F}});  // [2, 2]

    auto op = std::make_shared<MatMulOperation<float>>();
    auto result = op->forward({x, y});  // [2, 2]

    EXPECT_EQ(result.shape(), Shape({2, 2}));
    // [1*5 + 2*7, 1*6 + 2*8]   = [19, 22]
    // [3*5 + 4*7, 3*6 + 4*8]   = [43, 50]
    EXPECT_FLOAT_EQ((result[{0, 0}]), 19.0F);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 22.0F);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 43.0F);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 50.0F);
}

TEST_F(OpsGradTest, MatMulNumericalGradient) {
    auto x = Tensor<float>({{1.0F, 2.0F}, {3.0F, 4.0F}});
    auto y = Tensor<float>({{5.0F, 6.0F}, {7.0F, 8.0F}});

    auto op = std::make_shared<MatMulOperation<float>>();
    bool passed = ops::test::checkNumericalGradient(*op, {x, y});
    EXPECT_TRUE(passed);
}

// ========================================
// ExpOperation Tests
// ========================================

TEST_F(OpsGradTest, ExpForward) {
    auto x = Tensor<float>({0.0F, 1.0F, 2.0F});

    auto op = std::make_shared<ExpOperation<float>>();
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({3}));
    EXPECT_TRUE(approx_equal(result[{0}], std::exp(0.0F)));
    EXPECT_TRUE(approx_equal(result[{1}], std::exp(1.0F)));
    EXPECT_TRUE(approx_equal(result[{2}], std::exp(2.0F)));
}

TEST_F(OpsGradTest, ExpNumericalGradient) {
    auto x = Tensor<float>({0.0F, 1.0F, 2.0F});

    auto op = std::make_shared<ExpOperation<float>>();
    bool passed = ops::test::checkNumericalGradient(*op, {x});
    EXPECT_TRUE(passed);
}

// ========================================
// LogOperation Tests
// ========================================

TEST_F(OpsGradTest, LogForward) {
    auto x = Tensor<float>({1.0F, 2.0F, 10.0F});

    auto op = std::make_shared<LogOperation<float>>();
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({3}));
    EXPECT_TRUE(approx_equal(result[{0}], std::log(1.0F)));
    EXPECT_TRUE(approx_equal(result[{1}], std::log(2.0F)));
    EXPECT_TRUE(approx_equal(result[{2}], std::log(10.0F)));
}

TEST_F(OpsGradTest, LogNumericalGradient) {
    auto x = Tensor<float>({1.0F, 2.0F, 10.0F});

    auto op = std::make_shared<LogOperation<float>>();
    bool passed = ops::test::checkNumericalGradient(*op, {x});
    EXPECT_TRUE(passed);
}

// ========================================
// SqrtOperation Tests
// ========================================

TEST_F(OpsGradTest, SqrtForward) {
    auto x = Tensor<float>({1.0F, 4.0F, 9.0F, 16.0F});

    auto op = std::make_shared<SqrtOperation<float>>();
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({4}));
    EXPECT_FLOAT_EQ(result[{0}], 1.0F);
    EXPECT_FLOAT_EQ(result[{1}], 2.0F);
    EXPECT_FLOAT_EQ(result[{2}], 3.0F);
    EXPECT_FLOAT_EQ(result[{3}], 4.0F);
}

TEST_F(OpsGradTest, SqrtNumericalGradient) {
    auto x = Tensor<float>({1.0F, 4.0F, 9.0F});

    auto op = std::make_shared<SqrtOperation<float>>();
    bool passed = ops::test::checkNumericalGradient(*op, {x});
    EXPECT_TRUE(passed);
}

// ========================================
// PowOperation Tests
// ========================================

TEST_F(OpsGradTest, PowForward) {
    auto x = Tensor<float>({2.0F, 3.0F, 4.0F});
    auto y = Tensor<float>({2.0F, 3.0F, 2.0F});

    auto op = std::make_shared<PowOperation<float>>();
    auto result = op->forward({x, y});

    EXPECT_EQ(result.shape(), Shape({3}));
    EXPECT_FLOAT_EQ(result[{0}], 4.0F);   // 2^2
    EXPECT_FLOAT_EQ(result[{1}], 27.0F);  // 3^3
    EXPECT_FLOAT_EQ(result[{2}], 16.0F);  // 4^2
}

TEST_F(OpsGradTest, PowNumericalGradient) {
    auto x = Tensor<float>({2.0F, 3.0F});
    auto y = Tensor<float>({2.0F, 3.0F});

    auto op = std::make_shared<PowOperation<float>>();
    bool passed = ops::test::checkNumericalGradient(*op, {x, y});
    EXPECT_TRUE(passed);
}
