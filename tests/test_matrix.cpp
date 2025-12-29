#include <gtest/gtest.h>

#include <gradflow/matrix.hpp>

namespace gradflow {
namespace test {

class MatrixTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Set random seed for reproducibility
        std::srand(42);
    }
};

TEST_F(MatrixTest, DefaultConstruction) {
    Matrix<double> mat;
    EXPECT_EQ(mat.rows(), 0);
    EXPECT_EQ(mat.cols(), 0);
}

TEST_F(MatrixTest, SizeConstruction) {
    Matrix<double> mat(3, 4);
    EXPECT_EQ(mat.rows(), 3);
    EXPECT_EQ(mat.cols(), 4);
}

TEST_F(MatrixTest, InitializerListConstruction) {
    Matrix<double> mat = {{1.0, 2.0}, {3.0, 4.0}};
    EXPECT_EQ(mat.rows(), 2);
    EXPECT_EQ(mat.cols(), 2);
    EXPECT_DOUBLE_EQ(mat(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(mat(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(mat(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(mat(1, 1), 4.0);
}

TEST_F(MatrixTest, ElementAccess) {
    Matrix<double> mat(2, 2);
    mat(0, 0) = 1.0;
    mat(0, 1) = 2.0;
    mat(1, 0) = 3.0;
    mat(1, 1) = 4.0;

    EXPECT_DOUBLE_EQ(mat(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(mat(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(mat(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(mat(1, 1), 4.0);
}

TEST_F(MatrixTest, Addition) {
    Matrix<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    Matrix<double> b = {{5.0, 6.0}, {7.0, 8.0}};
    Matrix<double> c = a + b;

    EXPECT_DOUBLE_EQ(c(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(c(0, 1), 8.0);
    EXPECT_DOUBLE_EQ(c(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(c(1, 1), 12.0);
}

TEST_F(MatrixTest, Multiplication) {
    Matrix<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    Matrix<double> b = {{5.0, 6.0}, {7.0, 8.0}};
    Matrix<double> c = a * b;

    EXPECT_DOUBLE_EQ(c(0, 0), 19.0);  // 1*5 + 2*7
    EXPECT_DOUBLE_EQ(c(0, 1), 22.0);  // 1*6 + 2*8
    EXPECT_DOUBLE_EQ(c(1, 0), 43.0);  // 3*5 + 4*7
    EXPECT_DOUBLE_EQ(c(1, 1), 50.0);  // 3*6 + 4*8
}

TEST_F(MatrixTest, Transpose) {
    Matrix<double> a = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    Matrix<double> b = a.transpose();

    EXPECT_EQ(b.rows(), 3);
    EXPECT_EQ(b.cols(), 2);
    EXPECT_DOUBLE_EQ(b(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(b(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(b(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(b(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(b(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(b(2, 1), 6.0);
}

TEST_F(MatrixTest, ScalarMultiplication) {
    Matrix<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    Matrix<double> b = a * 2.0;

    EXPECT_DOUBLE_EQ(b(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(b(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(b(1, 0), 6.0);
    EXPECT_DOUBLE_EQ(b(1, 1), 8.0);
}

TEST_F(MatrixTest, Zeros) {
    Matrix<double> mat = Matrix<double>::zeros(3, 4);
    EXPECT_EQ(mat.rows(), 3);
    EXPECT_EQ(mat.cols(), 4);

    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            EXPECT_DOUBLE_EQ(mat(i, j), 0.0);
        }
    }
}

TEST_F(MatrixTest, Ones) {
    Matrix<double> mat = Matrix<double>::ones(2, 3);
    EXPECT_EQ(mat.rows(), 2);
    EXPECT_EQ(mat.cols(), 3);

    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            EXPECT_DOUBLE_EQ(mat(i, j), 1.0);
        }
    }
}

TEST_F(MatrixTest, Identity) {
    Matrix<double> mat = Matrix<double>::identity(3);
    EXPECT_EQ(mat.rows(), 3);
    EXPECT_EQ(mat.cols(), 3);

    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            if (i == j) {
                EXPECT_DOUBLE_EQ(mat(i, j), 1.0);
            } else {
                EXPECT_DOUBLE_EQ(mat(i, j), 0.0);
            }
        }
    }
}

}  // namespace test
}  // namespace gradflow
