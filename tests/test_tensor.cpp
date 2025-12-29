#include "gradflow/autograd/tensor.hpp"

#include <gtest/gtest.h>

using namespace gradflow;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ========================================
// Construction Tests
// ========================================

TEST_F(TensorTest, ConstructionDefault) {
    Tensor<float> t;

    EXPECT_EQ(t.ndim(), 0);
    EXPECT_EQ(t.size(), 1);  // Scalar has size 1
    EXPECT_EQ(t.data(), nullptr);
}

TEST_F(TensorTest, ConstructionWithShape) {
    Shape shape({2, 3});
    Tensor<float> t(shape);

    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.size(), 6);
    EXPECT_NE(t.data(), nullptr);
    EXPECT_TRUE(t.is_contiguous());
}

TEST_F(TensorTest, ConstructionFromInitializerList1D) {
    Tensor<float> t({1.0f, 2.0f, 3.0f, 4.0f});

    EXPECT_EQ(t.ndim(), 1);
    EXPECT_EQ(t.size(), 4);
    EXPECT_FLOAT_EQ((t[{0}]), 1.0f);
    EXPECT_FLOAT_EQ((t[{1}]), 2.0f);
    EXPECT_FLOAT_EQ((t[{2}]), 3.0f);
    EXPECT_FLOAT_EQ((t[{3}]), 4.0f);
}

TEST_F(TensorTest, ConstructionFromInitializerList2D) {
    Tensor<float> t({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});

    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.shape(), Shape({2, 3}));
    EXPECT_EQ(t.size(), 6);
    EXPECT_FLOAT_EQ((t[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((t[{0, 1}]), 2.0f);
    EXPECT_FLOAT_EQ((t[{0, 2}]), 3.0f);
    EXPECT_FLOAT_EQ((t[{1, 0}]), 4.0f);
    EXPECT_FLOAT_EQ((t[{1, 1}]), 5.0f);
    EXPECT_FLOAT_EQ((t[{1, 2}]), 6.0f);
}

TEST_F(TensorTest, ConstructionFromShapeAndData) {
    Shape shape({2, 3});
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor<float> t(shape, data);

    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.size(), 6);
    EXPECT_FLOAT_EQ((t[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((t[{1, 2}]), 6.0f);
}

TEST_F(TensorTest, ConstructionFromShapeAndDataMismatch) {
    Shape shape({2, 3});
    std::vector<float> data = {1.0f, 2.0f};  // Wrong size

    EXPECT_THROW(Tensor<float> t(shape, data), std::invalid_argument);
}

TEST_F(TensorTest, ConstructionCopyAndMove) {
    Tensor<float> t1({1.0f, 2.0f, 3.0f});

    // Copy constructor
    Tensor<float> t2(t1);
    EXPECT_EQ(t2.size(), 3);
    EXPECT_FLOAT_EQ(t2[{0}], 1.0f);

    // Move constructor
    Tensor<float> t3(std::move(t1));
    EXPECT_EQ(t3.size(), 3);
    EXPECT_FLOAT_EQ(t3[{0}], 1.0f);
}

// ========================================
// Element Access Tests
// ========================================

TEST_F(TensorTest, ElementAccess) {
    Tensor<float> t({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});

    // Read access
    EXPECT_FLOAT_EQ((t[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((t[{0, 1}]), 2.0f);
    EXPECT_FLOAT_EQ((t[{1, 0}]), 4.0f);
    EXPECT_FLOAT_EQ((t[{1, 2}]), 6.0f);

    // Write access
    t[{0, 0}] = 10.0f;
    t[{1, 2}] = 20.0f;
    EXPECT_FLOAT_EQ((t[{0, 0}]), 10.0f);
    EXPECT_FLOAT_EQ((t[{1, 2}]), 20.0f);
}

TEST_F(TensorTest, ElementAccessOutOfBounds) {
    Tensor<float> t({{1.0f, 2.0f}, {3.0f, 4.0f}});

    EXPECT_THROW((t[{2, 0}]), std::out_of_range);
    EXPECT_THROW((t[{0, 2}]), std::out_of_range);
    EXPECT_THROW((t[{0}]), std::out_of_range);  // Wrong number of indices
}

// ========================================
// Reshape Tests
// ========================================

TEST_F(TensorTest, ReshapeContiguous) {
    Tensor<float> t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    EXPECT_TRUE(t.is_contiguous());

    auto reshaped = t.reshape(Shape({2, 3}));

    EXPECT_EQ(reshaped.shape(), Shape({2, 3}));
    EXPECT_EQ(reshaped.size(), 6);
    EXPECT_FLOAT_EQ((reshaped[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((reshaped[{0, 1}]), 2.0f);
    EXPECT_FLOAT_EQ((reshaped[{1, 2}]), 6.0f);
    EXPECT_TRUE(reshaped.is_contiguous());
}

TEST_F(TensorTest, ReshapeNonContiguous) {
    Tensor<float> t({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    auto transposed = t.transpose(0, 1);
    EXPECT_FALSE(transposed.is_contiguous());

    auto reshaped = transposed.reshape(Shape({6}));

    EXPECT_EQ(reshaped.shape(), Shape({6}));
    EXPECT_TRUE(reshaped.is_contiguous());
}

TEST_F(TensorTest, ReshapeSizeMismatch) {
    Tensor<float> t({1.0f, 2.0f, 3.0f, 4.0f});

    EXPECT_THROW(t.reshape(Shape({2, 3})), std::invalid_argument);
}

TEST_F(TensorTest, ViewContiguous) {
    Tensor<float> t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    EXPECT_TRUE(t.is_contiguous());

    auto viewed = t.view(Shape({3, 2}));

    EXPECT_EQ(viewed.shape(), Shape({3, 2}));
    EXPECT_FLOAT_EQ((viewed[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((viewed[{0, 1}]), 2.0f);
    EXPECT_FLOAT_EQ((viewed[{2, 1}]), 6.0f);

    // Modifying view should affect original
    viewed[{0, 0}] = 100.0f;
    EXPECT_FLOAT_EQ((t[{0}]), 100.0f);
}

TEST_F(TensorTest, ViewNonContiguous) {
    Tensor<float> t({{1.0f, 2.0f}, {3.0f, 4.0f}});
    auto transposed = t.transpose(0, 1);
    EXPECT_FALSE(transposed.is_contiguous());

    EXPECT_THROW(transposed.view(Shape({4})), std::invalid_argument);
}

// ========================================
// Transpose Tests
// ========================================

TEST_F(TensorTest, Transpose2D) {
    Tensor<float> t({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});

    auto transposed = t.transpose(0, 1);

    EXPECT_EQ(transposed.shape(), Shape({3, 2}));
    EXPECT_FLOAT_EQ((transposed[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((transposed[{0, 1}]), 4.0f);
    EXPECT_FLOAT_EQ((transposed[{1, 0}]), 2.0f);
    EXPECT_FLOAT_EQ((transposed[{1, 1}]), 5.0f);
    EXPECT_FLOAT_EQ((transposed[{2, 0}]), 3.0f);
    EXPECT_FLOAT_EQ((transposed[{2, 1}]), 6.0f);

    // Transposed view is not contiguous
    EXPECT_FALSE(transposed.is_contiguous());
}

TEST_F(TensorTest, TransposeZeroCopy) {
    Tensor<float> t({{1.0f, 2.0f}, {3.0f, 4.0f}});

    auto transposed = t.transpose(0, 1);

    // Modifying transposed view should affect original
    transposed[{0, 1}] = 100.0f;
    EXPECT_FLOAT_EQ((t[{1, 0}]), 100.0f);
}

TEST_F(TensorTest, TransposeSameDimension) {
    Tensor<float> t({{1.0f, 2.0f}, {3.0f, 4.0f}});

    auto transposed = t.transpose(0, 0);

    EXPECT_EQ(transposed.shape(), t.shape());
}

TEST_F(TensorTest, TransposeOutOfBounds) {
    Tensor<float> t({{1.0f, 2.0f}, {3.0f, 4.0f}});

    EXPECT_THROW(t.transpose(0, 2), std::out_of_range);
}

TEST_F(TensorTest, PermuteBasic) {
    Tensor<float> t(Shape({2, 3, 4}));

    auto permuted = t.permute({2, 0, 1});

    EXPECT_EQ(permuted.shape(), Shape({4, 2, 3}));
}

TEST_F(TensorTest, PermuteInvalid) {
    Tensor<float> t(Shape({2, 3, 4}));

    EXPECT_THROW(t.permute({0, 1}), std::invalid_argument);     // Wrong size
    EXPECT_THROW(t.permute({0, 0, 1}), std::invalid_argument);  // Duplicate
    EXPECT_THROW(t.permute({0, 1, 5}), std::invalid_argument);  // Out of range
}

// ========================================
// Slicing Tests
// ========================================

TEST_F(TensorTest, SlicingBasic) {
    Tensor<float> t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    auto sliced = t.slice(0, 1, 4);

    EXPECT_EQ(sliced.shape(), Shape({3}));
    EXPECT_FLOAT_EQ((sliced[{0}]), 2.0f);
    EXPECT_FLOAT_EQ((sliced[{1}]), 3.0f);
    EXPECT_FLOAT_EQ((sliced[{2}]), 4.0f);
}

TEST_F(TensorTest, Slicing2D) {
    Tensor<float> t({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}});

    // Slice rows
    auto row_slice = t.slice(0, 1, 3);
    EXPECT_EQ(row_slice.shape(), Shape({2, 3}));
    EXPECT_FLOAT_EQ((row_slice[{0, 0}]), 4.0f);
    EXPECT_FLOAT_EQ((row_slice[{1, 2}]), 9.0f);

    // Slice columns
    auto col_slice = t.slice(1, 0, 2);
    EXPECT_EQ(col_slice.shape(), Shape({3, 2}));
    EXPECT_FLOAT_EQ((col_slice[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((col_slice[{0, 1}]), 2.0f);
    EXPECT_FLOAT_EQ((col_slice[{2, 1}]), 8.0f);
}

TEST_F(TensorTest, SlicingZeroCopy) {
    Tensor<float> t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    auto sliced = t.slice(0, 1, 4);

    // Modifying slice should affect original
    sliced[{0}] = 100.0f;
    EXPECT_FLOAT_EQ((t[{1}]), 100.0f);
}

TEST_F(TensorTest, SlicingOutOfBounds) {
    Tensor<float> t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    EXPECT_THROW(t.slice(1, 0, 2), std::out_of_range);   // Dimension out of range
    EXPECT_THROW(t.slice(0, 3, 2), std::out_of_range);   // start >= end
    EXPECT_THROW(t.slice(0, 0, 10), std::out_of_range);  // end > size
}

// ========================================
// View Zero-Copy Tests
// ========================================

TEST_F(TensorTest, ViewZeroCopy) {
    Tensor<float> t({1.0f, 2.0f, 3.0f, 4.0f});

    auto viewed = t.view(Shape({2, 2}));

    // Verify data is shared (zero-copy)
    EXPECT_EQ(t.data(), viewed.data());

    // Modifying view affects original
    viewed[{0, 1}] = 100.0f;
    EXPECT_FLOAT_EQ((t[{1}]), 100.0f);

    // Modifying original affects view
    t[{3}] = 200.0f;
    EXPECT_FLOAT_EQ((viewed[{1, 1}]), 200.0f);
}

TEST_F(TensorTest, TransposeZeroCopyVerification) {
    Tensor<float> t({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});

    auto transposed = t.transpose(0, 1);

    // Verify data pointer is the same (zero-copy)
    EXPECT_EQ(t.data(), transposed.data());
}

TEST_F(TensorTest, SliceZeroCopyVerification) {
    Tensor<float> t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    auto sliced = t.slice(0, 1, 4);

    // Verify data is shared (may have different pointer due to offset)
    // But modifying one affects the other
    sliced[{0}] = 999.0f;
    EXPECT_FLOAT_EQ((t[{1}]), 999.0f);
}

// ========================================
// Contiguous Tests
// ========================================

TEST_F(TensorTest, ContiguousCheck) {
    // Newly created tensor is contiguous
    Tensor<float> t1({1.0f, 2.0f, 3.0f, 4.0f});
    EXPECT_TRUE(t1.is_contiguous());

    // View is contiguous
    auto viewed = t1.view(Shape({2, 2}));
    EXPECT_TRUE(viewed.is_contiguous());

    // Transposed tensor is not contiguous
    Tensor<float> t2({{1.0f, 2.0f}, {3.0f, 4.0f}});
    auto transposed = t2.transpose(0, 1);
    EXPECT_FALSE(transposed.is_contiguous());

    // Sliced tensor may not be contiguous
    Tensor<float> t3({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    auto col_slice = t3.slice(1, 0, 2);
    EXPECT_FALSE(col_slice.is_contiguous());
}

TEST_F(TensorTest, ContiguousOperation) {
    Tensor<float> t({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    auto transposed = t.transpose(0, 1);
    EXPECT_FALSE(transposed.is_contiguous());

    // Make it contiguous
    auto contiguous = transposed.contiguous();

    EXPECT_TRUE(contiguous.is_contiguous());
    EXPECT_EQ(contiguous.shape(), Shape({3, 2}));
    EXPECT_FLOAT_EQ((contiguous[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((contiguous[{0, 1}]), 4.0f);
    EXPECT_FLOAT_EQ((contiguous[{2, 1}]), 6.0f);

    // Original should not be affected
    EXPECT_FALSE(transposed.is_contiguous());
}

TEST_F(TensorTest, ContiguousAlreadyContiguous) {
    Tensor<float> t({1.0f, 2.0f, 3.0f, 4.0f});
    EXPECT_TRUE(t.is_contiguous());

    auto contiguous = t.contiguous();

    // Should return a copy
    EXPECT_TRUE(contiguous.is_contiguous());
    EXPECT_EQ(contiguous.size(), 4);
}

// ========================================
// Factory Function Tests
// ========================================

TEST_F(TensorTest, FactoryZeros) {
    auto t = Tensor<float>::zeros(Shape({2, 3}));

    EXPECT_EQ(t.shape(), Shape({2, 3}));
    EXPECT_EQ(t.size(), 6);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ((t[{i, j}]), 0.0f);
        }
    }
}

TEST_F(TensorTest, FactoryOnes) {
    auto t = Tensor<float>::ones(Shape({2, 3}));

    EXPECT_EQ(t.shape(), Shape({2, 3}));
    EXPECT_EQ(t.size(), 6);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ((t[{i, j}]), 1.0f);
        }
    }
}

TEST_F(TensorTest, FactoryRandn) {
    auto t = Tensor<float>::randn(Shape({100}));

    EXPECT_EQ(t.shape(), Shape({100}));
    EXPECT_EQ(t.size(), 100);

    // Check that values are not all the same (probabilistic test)
    float first_value = t[{0}];
    bool has_different_value = false;
    for (size_t i = 1; i < 100; ++i) {
        if (std::abs(t[{i}] - first_value) > 0.001f) {
            has_different_value = true;
            break;
        }
    }
    EXPECT_TRUE(has_different_value);
}

TEST_F(TensorTest, FactoryRand) {
    auto t = Tensor<float>::rand(Shape({100}));

    EXPECT_EQ(t.shape(), Shape({100}));
    EXPECT_EQ(t.size(), 100);

    // Check that all values are in [0, 1)
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_GE(t[{i}], 0.0f);
        EXPECT_LT(t[{i}], 1.0f);
    }
}

TEST_F(TensorTest, FactoryEye) {
    auto t = Tensor<float>::eye(3);

    EXPECT_EQ(t.shape(), Shape({3, 3}));
    EXPECT_EQ(t.size(), 9);

    // Check diagonal is 1
    EXPECT_FLOAT_EQ((t[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((t[{1, 1}]), 1.0f);
    EXPECT_FLOAT_EQ((t[{2, 2}]), 1.0f);

    // Check off-diagonal is 0
    EXPECT_FLOAT_EQ((t[{0, 1}]), 0.0f);
    EXPECT_FLOAT_EQ((t[{0, 2}]), 0.0f);
    EXPECT_FLOAT_EQ((t[{1, 0}]), 0.0f);
    EXPECT_FLOAT_EQ((t[{1, 2}]), 0.0f);
    EXPECT_FLOAT_EQ((t[{2, 0}]), 0.0f);
    EXPECT_FLOAT_EQ((t[{2, 1}]), 0.0f);
}

TEST_F(TensorTest, FactoryZerosLike) {
    Tensor<float> t({{1.0f, 2.0f}, {3.0f, 4.0f}});

    auto zeros = Tensor<float>::zeros_like(t);

    EXPECT_EQ(zeros.shape(), t.shape());
    EXPECT_EQ(zeros.device(), t.device());
    EXPECT_FLOAT_EQ((zeros[{0, 0}]), 0.0f);
    EXPECT_FLOAT_EQ((zeros[{1, 1}]), 0.0f);
}

TEST_F(TensorTest, FactoryOnesLike) {
    Tensor<float> t({{1.0f, 2.0f}, {3.0f, 4.0f}});

    auto ones = Tensor<float>::ones_like(t);

    EXPECT_EQ(ones.shape(), t.shape());
    EXPECT_EQ(ones.device(), t.device());
    EXPECT_FLOAT_EQ((ones[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((ones[{1, 1}]), 1.0f);
}
