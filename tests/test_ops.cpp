#include <gtest/gtest.h>

#include <cmath>

#include "gradflow/autograd/ops/elementwise.hpp"
#include "gradflow/autograd/ops/matmul.hpp"
#include "gradflow/autograd/ops/reduction.hpp"

using namespace gradflow;

class OpsTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}

  // Helper function to check if two floats are approximately equal
  bool approx_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
  }
};

// ========================================
// Add Operation Tests (Red Phase)
// ========================================

TEST_F(OpsTest, AddBasic) {
  Tensor<float> a({1.0F, 2.0F, 3.0F});
  Tensor<float> b({4.0F, 5.0F, 6.0F});

  auto c = add(a, b);

  EXPECT_EQ(c.shape(), Shape({3}));
  EXPECT_FLOAT_EQ(c[{0}], 5.0F);
  EXPECT_FLOAT_EQ(c[{1}], 7.0F);
  EXPECT_FLOAT_EQ(c[{2}], 9.0F);
}

TEST_F(OpsTest, Add2D) {
  Tensor<float> a({{1.0F, 2.0F}, {3.0F, 4.0F}});
  Tensor<float> b({{5.0F, 6.0F}, {7.0F, 8.0F}});

  auto c = add(a, b);

  EXPECT_EQ(c.shape(), Shape({2, 2}));
  EXPECT_FLOAT_EQ((c[{0, 0}]), 6.0F);
  EXPECT_FLOAT_EQ((c[{0, 1}]), 8.0F);
  EXPECT_FLOAT_EQ((c[{1, 0}]), 10.0F);
  EXPECT_FLOAT_EQ((c[{1, 1}]), 12.0F);
}

// ========================================
// Subtract Operation Tests (Red Phase)
// ========================================

TEST_F(OpsTest, SubBasic) {
  Tensor<float> a({5.0F, 7.0F, 9.0F});
  Tensor<float> b({1.0F, 2.0F, 3.0F});

  auto c = sub(a, b);

  EXPECT_EQ(c.shape(), Shape({3}));
  EXPECT_FLOAT_EQ(c[{0}], 4.0F);
  EXPECT_FLOAT_EQ(c[{1}], 5.0F);
  EXPECT_FLOAT_EQ(c[{2}], 6.0F);
}

// ========================================
// Multiply Operation Tests (Red Phase)
// ========================================

TEST_F(OpsTest, MulBasic) {
  Tensor<float> a({2.0F, 3.0F, 4.0F});
  Tensor<float> b({5.0F, 6.0F, 7.0F});

  auto c = mul(a, b);

  EXPECT_EQ(c.shape(), Shape({3}));
  EXPECT_FLOAT_EQ(c[{0}], 10.0F);
  EXPECT_FLOAT_EQ(c[{1}], 18.0F);
  EXPECT_FLOAT_EQ(c[{2}], 28.0F);
}

// ========================================
// Divide Operation Tests (Red Phase)
// ========================================

TEST_F(OpsTest, DivBasic) {
  Tensor<float> a({10.0F, 20.0F, 30.0F});
  Tensor<float> b({2.0F, 4.0F, 5.0F});

  auto c = div(a, b);

  EXPECT_EQ(c.shape(), Shape({3}));
  EXPECT_FLOAT_EQ(c[{0}], 5.0F);
  EXPECT_FLOAT_EQ(c[{1}], 5.0F);
  EXPECT_FLOAT_EQ(c[{2}], 6.0F);
}

// ========================================
// Broadcasting Tests (Red Phase)
// ========================================

TEST_F(OpsTest, AddBroadcastScalarTo1D) {
  Tensor<float> a({1.0F, 2.0F, 3.0F});
  Tensor<float> b({10.0F});

  auto c = add(a, b);

  EXPECT_EQ(c.shape(), Shape({3}));
  EXPECT_FLOAT_EQ(c[{0}], 11.0F);
  EXPECT_FLOAT_EQ(c[{1}], 12.0F);
  EXPECT_FLOAT_EQ(c[{2}], 13.0F);
}

TEST_F(OpsTest, AddBroadcast1DTo2D) {
  Tensor<float> a({{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}});  // [2, 3]
  Tensor<float> b({10.0F, 20.0F, 30.0F});                     // [3]

  auto c = add(a, b);

  EXPECT_EQ(c.shape(), Shape({2, 3}));
  EXPECT_FLOAT_EQ((c[{0, 0}]), 11.0F);
  EXPECT_FLOAT_EQ((c[{0, 1}]), 22.0F);
  EXPECT_FLOAT_EQ((c[{0, 2}]), 33.0F);
  EXPECT_FLOAT_EQ((c[{1, 0}]), 14.0F);
  EXPECT_FLOAT_EQ((c[{1, 1}]), 25.0F);
  EXPECT_FLOAT_EQ((c[{1, 2}]), 36.0F);
}

// ========================================
// Shape Mismatch Tests (Red Phase)
// ========================================

TEST_F(OpsTest, AddShapeMismatch) {
  Tensor<float> a({1.0F, 2.0F, 3.0F});
  Tensor<float> b({1.0F, 2.0F});

  EXPECT_THROW(add(a, b), std::invalid_argument);
}

// ========================================
// Matrix Multiplication Tests (Red Phase)
// ========================================

TEST_F(OpsTest, MatMulBasic) {
  Tensor<float> a({{1.0F, 2.0F}, {3.0F, 4.0F}});  // [2, 2]
  Tensor<float> b({{5.0F, 6.0F}, {7.0F, 8.0F}});  // [2, 2]

  auto c = matmul(a, b);  // [2, 2]

  EXPECT_EQ(c.shape(), Shape({2, 2}));
  // [1*5 + 2*7, 1*6 + 2*8]   = [19, 22]
  // [3*5 + 4*7, 3*6 + 4*8]   = [43, 50]
  EXPECT_FLOAT_EQ((c[{0, 0}]), 19.0F);
  EXPECT_FLOAT_EQ((c[{0, 1}]), 22.0F);
  EXPECT_FLOAT_EQ((c[{1, 0}]), 43.0F);
  EXPECT_FLOAT_EQ((c[{1, 1}]), 50.0F);
}

TEST_F(OpsTest, MatMulRectangular) {
  Tensor<float> a({{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}});       // [2, 3]
  Tensor<float> b({{7.0F, 8.0F}, {9.0F, 10.0F}, {11.0F, 12.0F}});  // [3, 2]

  auto c = matmul(a, b);  // [2, 2]

  EXPECT_EQ(c.shape(), Shape({2, 2}));
  // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12]   = [58, 64]
  // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]   = [139, 154]
  EXPECT_FLOAT_EQ((c[{0, 0}]), 58.0F);
  EXPECT_FLOAT_EQ((c[{0, 1}]), 64.0F);
  EXPECT_FLOAT_EQ((c[{1, 0}]), 139.0F);
  EXPECT_FLOAT_EQ((c[{1, 1}]), 154.0F);
}

TEST_F(OpsTest, MatMulIncompatibleShapes) {
  Tensor<float> a({{1.0F, 2.0F}, {3.0F, 4.0F}});                 // [2, 2]
  Tensor<float> b({{5.0F, 6.0F}, {7.0F, 8.0F}, {9.0F, 10.0F}});  // [3, 2]

  EXPECT_THROW(matmul(a, b), std::invalid_argument);
}

// ========================================
// Reduction Operation Tests (Red Phase)
// ========================================

TEST_F(OpsTest, SumAllElements) {
  Tensor<float> a({{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}});

  auto result = sum(a);

  EXPECT_EQ(result.shape(), Shape({}));  // scalar
  EXPECT_FLOAT_EQ(result[{}], 21.0F);
}

TEST_F(OpsTest, SumAlongAxis) {
  Tensor<float> a({{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}});

  // Sum along axis 0 (reduce rows)
  auto result0 = sum(a, 0);
  EXPECT_EQ(result0.shape(), Shape({3}));
  EXPECT_FLOAT_EQ(result0[{0}], 5.0F);  // 1 + 4
  EXPECT_FLOAT_EQ(result0[{1}], 7.0F);  // 2 + 5
  EXPECT_FLOAT_EQ(result0[{2}], 9.0F);  // 3 + 6

  // Sum along axis 1 (reduce columns)
  auto result1 = sum(a, 1);
  EXPECT_EQ(result1.shape(), Shape({2}));
  EXPECT_FLOAT_EQ(result1[{0}], 6.0F);   // 1 + 2 + 3
  EXPECT_FLOAT_EQ(result1[{1}], 15.0F);  // 4 + 5 + 6
}

TEST_F(OpsTest, MeanAllElements) {
  Tensor<float> a({{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}});

  auto result = mean(a);

  EXPECT_EQ(result.shape(), Shape({}));
  EXPECT_FLOAT_EQ(result[{}], 3.5F);  // 21 / 6
}

TEST_F(OpsTest, MaxAllElements) {
  Tensor<float> a({{1.0F, 5.0F, 3.0F}, {4.0F, 2.0F, 6.0F}});

  auto result = max(a);

  EXPECT_EQ(result.shape(), Shape({}));
  EXPECT_FLOAT_EQ(result[{}], 6.0F);
}

TEST_F(OpsTest, MinAllElements) {
  Tensor<float> a({{1.0F, 5.0F, 3.0F}, {4.0F, 2.0F, 6.0F}});

  auto result = min(a);

  EXPECT_EQ(result.shape(), Shape({}));
  EXPECT_FLOAT_EQ(result[{}], 1.0F);
}

// ========================================
// Mathematical Function Tests
// ========================================

TEST_F(OpsTest, ExpFunction) {
  Tensor<float> a({0.0F, 1.0F, 2.0F});

  auto result = exp(a);

  EXPECT_EQ(result.shape(), Shape({3}));
  EXPECT_TRUE(approx_equal(result[{0}], std::exp(0.0F)));
  EXPECT_TRUE(approx_equal(result[{1}], std::exp(1.0F)));
  EXPECT_TRUE(approx_equal(result[{2}], std::exp(2.0F)));
}

TEST_F(OpsTest, LogFunction) {
  Tensor<float> a({1.0F, 2.0F, 10.0F});

  auto result = log(a);

  EXPECT_EQ(result.shape(), Shape({3}));
  EXPECT_TRUE(approx_equal(result[{0}], std::log(1.0F)));
  EXPECT_TRUE(approx_equal(result[{1}], std::log(2.0F)));
  EXPECT_TRUE(approx_equal(result[{2}], std::log(10.0F)));
}

TEST_F(OpsTest, SqrtFunction) {
  Tensor<float> a({1.0F, 4.0F, 9.0F, 16.0F});

  auto result = sqrt(a);

  EXPECT_EQ(result.shape(), Shape({4}));
  EXPECT_FLOAT_EQ(result[{0}], 1.0F);
  EXPECT_FLOAT_EQ(result[{1}], 2.0F);
  EXPECT_FLOAT_EQ(result[{2}], 3.0F);
  EXPECT_FLOAT_EQ(result[{3}], 4.0F);
}

TEST_F(OpsTest, PowFunction) {
  Tensor<float> a({2.0F, 3.0F, 4.0F});
  Tensor<float> b({2.0F, 3.0F, 2.0F});

  auto result = pow(a, b);

  EXPECT_EQ(result.shape(), Shape({3}));
  EXPECT_FLOAT_EQ(result[{0}], 4.0F);   // 2^2
  EXPECT_FLOAT_EQ(result[{1}], 27.0F);  // 3^3
  EXPECT_FLOAT_EQ(result[{2}], 16.0F);  // 4^2
}

TEST_F(OpsTest, PowScalarExponent) {
  Tensor<float> a({2.0F, 3.0F, 4.0F});

  auto result = pow(a, 3.0F);

  EXPECT_EQ(result.shape(), Shape({3}));
  EXPECT_FLOAT_EQ(result[{0}], 8.0F);   // 2^3
  EXPECT_FLOAT_EQ(result[{1}], 27.0F);  // 3^3
  EXPECT_FLOAT_EQ(result[{2}], 64.0F);  // 4^3
}
