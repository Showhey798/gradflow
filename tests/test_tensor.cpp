#include "gradflow/autograd/tensor.hpp"

#include <gtest/gtest.h>

using gradflow::Shape;
using gradflow::Tensor;

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
    const Shape kShape({2, 3});
    Tensor<float> t(kShape);

    EXPECT_EQ(t.shape(), kShape);
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.size(), 6);
    EXPECT_NE(t.data(), nullptr);
    EXPECT_TRUE(t.isContiguous());
}

TEST_F(TensorTest, ConstructionFromInitializerList1D) {
    const Tensor<float> kT({1.0F, 2.0F, 3.0F, 4.0F});

    EXPECT_EQ(kT.ndim(), 1);
    EXPECT_EQ(kT.size(), 4);
    EXPECT_FLOAT_EQ((kT[{0}]), 1.0F);
    EXPECT_FLOAT_EQ((kT[{1}]), 2.0F);
    EXPECT_FLOAT_EQ((kT[{2}]), 3.0F);
    EXPECT_FLOAT_EQ((kT[{3}]), 4.0F);
}

TEST_F(TensorTest, ConstructionFromInitializerList2D) {
    Tensor<float> t({{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}});

    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.shape(), Shape({2, 3}));
    EXPECT_EQ(t.size(), 6);
    EXPECT_FLOAT_EQ((t[{0, 0}]), 1.0F);
    EXPECT_FLOAT_EQ((t[{0, 1}]), 2.0F);
    EXPECT_FLOAT_EQ((t[{0, 2}]), 3.0F);
    EXPECT_FLOAT_EQ((t[{1, 0}]), 4.0F);
    EXPECT_FLOAT_EQ((t[{1, 1}]), 5.0F);
    EXPECT_FLOAT_EQ((t[{1, 2}]), 6.0F);
}

TEST_F(TensorTest, ConstructionFromShapeAndData) {
    const Shape kShape({2, 3});
    const std::vector<float> kData = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    Tensor<float> t(kShape, kData);

    EXPECT_EQ(t.shape(), kShape);
    EXPECT_EQ(t.size(), 6);
    EXPECT_FLOAT_EQ((t[{0, 0}]), 1.0F);
    EXPECT_FLOAT_EQ((t[{1, 2}]), 6.0F);
}

TEST_F(TensorTest, ConstructionFromShapeAndDataMismatch) {
    const Shape kShape({2, 3});
    const std::vector<float> kData = {1.0F, 2.0F};  // Wrong size

    EXPECT_THROW(Tensor<float> t(kShape, kData), std::invalid_argument);
}

TEST_F(TensorTest, ConstructionCopyAndMove) {
    Tensor<float> t1({1.0F, 2.0F, 3.0F});

    // Copy constructor
    Tensor<float> t2(t1);
    EXPECT_EQ(t2.size(), 3);
    EXPECT_FLOAT_EQ(t2[{0}], 1.0F);

    // Move constructor
    Tensor<float> t3(std::move(t1));
    EXPECT_EQ(t3.size(), 3);
    EXPECT_FLOAT_EQ(t3[{0}], 1.0F);
}

// ========================================
// Element Access Tests
// ========================================

TEST_F(TensorTest, ElementAccess) {
    Tensor<float> t({{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}});

    // Read access
    EXPECT_FLOAT_EQ((t[{0, 0}]), 1.0F);
    EXPECT_FLOAT_EQ((t[{0, 1}]), 2.0F);
    EXPECT_FLOAT_EQ((t[{1, 0}]), 4.0F);
    EXPECT_FLOAT_EQ((t[{1, 2}]), 6.0F);

    // Write access
    t[{0, 0}] = 10.0F;
    t[{1, 2}] = 20.0F;
    EXPECT_FLOAT_EQ((t[{0, 0}]), 10.0F);
    EXPECT_FLOAT_EQ((t[{1, 2}]), 20.0F);
}

TEST_F(TensorTest, ElementAccessOutOfBounds) {
    Tensor<float> t({{1.0F, 2.0F}, {3.0F, 4.0F}});

    EXPECT_THROW((t[{2, 0}]), std::out_of_range);
    EXPECT_THROW((t[{0, 2}]), std::out_of_range);
    EXPECT_THROW((t[{0}]), std::out_of_range);  // Wrong number of indices
}

// ========================================
// Reshape Tests
// ========================================

TEST_F(TensorTest, ReshapeContiguous) {
    const Tensor<float> kT({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});

    EXPECT_TRUE(kT.isContiguous());

    auto reshaped = kT.reshape(Shape({2, 3}));

    EXPECT_EQ(reshaped.shape(), Shape({2, 3}));
    EXPECT_EQ(reshaped.size(), 6);
    EXPECT_FLOAT_EQ((reshaped[{0, 0}]), 1.0F);
    EXPECT_FLOAT_EQ((reshaped[{0, 1}]), 2.0F);
    EXPECT_FLOAT_EQ((reshaped[{1, 2}]), 6.0F);

    EXPECT_TRUE(reshaped.isContiguous());
}

TEST_F(TensorTest, ReshapeNonContiguous) {
    Tensor<float> t({{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}});
    auto transposed = t.transpose(0, 1);
    EXPECT_FALSE(transposed.isContiguous());

    auto reshaped = transposed.reshape(Shape({6}));

    EXPECT_EQ(reshaped.shape(), Shape({6}));
    EXPECT_TRUE(reshaped.isContiguous());
}

TEST_F(TensorTest, ReshapeSizeMismatch) {
    const Tensor<float> kT({1.0F, 2.0F, 3.0F, 4.0F});

    EXPECT_THROW(kT.reshape(Shape({2, 3})), std::invalid_argument);
}

TEST_F(TensorTest, ViewContiguous) {
    const Tensor<float> kT({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    EXPECT_TRUE(kT.isContiguous());

    auto viewed = kT.view(Shape({3, 2}));

    EXPECT_EQ(viewed.shape(), Shape({3, 2}));
    EXPECT_FLOAT_EQ((viewed[{0, 0}]), 1.0F);
    EXPECT_FLOAT_EQ((viewed[{0, 1}]), 2.0F);
    EXPECT_FLOAT_EQ((viewed[{2, 1}]), 6.0F);

    // Modifying view should affect original
    viewed[{0, 0}] = 100.0F;
    EXPECT_FLOAT_EQ((kT[{0}]), 100.0F);
}

TEST_F(TensorTest, ViewNonContiguous) {
    Tensor<float> t({{1.0F, 2.0F}, {3.0F, 4.0F}});
    auto transposed = t.transpose(0, 1);
    EXPECT_FALSE(transposed.isContiguous());

    EXPECT_THROW(transposed.view(Shape({4})), std::invalid_argument);
}

// ========================================
// Transpose Tests
// ========================================

TEST_F(TensorTest, Transpose2D) {
    Tensor<float> t({{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}});

    auto transposed = t.transpose(0, 1);

    EXPECT_EQ(transposed.shape(), Shape({3, 2}));
    EXPECT_FLOAT_EQ((transposed[{0, 0}]), 1.0F);
    EXPECT_FLOAT_EQ((transposed[{0, 1}]), 4.0F);
    EXPECT_FLOAT_EQ((transposed[{1, 0}]), 2.0F);
    EXPECT_FLOAT_EQ((transposed[{1, 1}]), 5.0F);
    EXPECT_FLOAT_EQ((transposed[{2, 0}]), 3.0F);
    EXPECT_FLOAT_EQ((transposed[{2, 1}]), 6.0F);

    // Transposed view is not contiguous
    EXPECT_FALSE(transposed.isContiguous());
}

TEST_F(TensorTest, TransposeZeroCopy) {
    Tensor<float> t({{1.0F, 2.0F}, {3.0F, 4.0F}});

    auto transposed = t.transpose(0, 1);

    // Modifying transposed view should affect original
    transposed[{0, 1}] = 100.0F;
    EXPECT_FLOAT_EQ((t[{1, 0}]), 100.0F);
}

TEST_F(TensorTest, TransposeSameDimension) {
    Tensor<float> t({{1.0F, 2.0F}, {3.0F, 4.0F}});

    auto transposed = t.transpose(0, 0);

    EXPECT_EQ(transposed.shape(), t.shape());
}

TEST_F(TensorTest, TransposeOutOfBounds) {
    Tensor<float> t({{1.0F, 2.0F}, {3.0F, 4.0F}});

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
    Tensor<float> t({1.0F, 2.0F, 3.0F, 4.0F, 5.0F});

    auto sliced = t.slice(0, 1, 4);

    EXPECT_EQ(sliced.shape(), Shape({3}));
    EXPECT_FLOAT_EQ((sliced[{0}]), 2.0F);
    EXPECT_FLOAT_EQ((sliced[{1}]), 3.0F);
    EXPECT_FLOAT_EQ((sliced[{2}]), 4.0F);
}

TEST_F(TensorTest, Slicing2D) {
    Tensor<float> t({{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}, {7.0F, 8.0F, 9.0F}});

    // Slice rows
    auto row_slice = t.slice(0, 1, 3);
    EXPECT_EQ(row_slice.shape(), Shape({2, 3}));
    EXPECT_FLOAT_EQ((row_slice[{0, 0}]), 4.0F);
    EXPECT_FLOAT_EQ((row_slice[{1, 2}]), 9.0F);

    // Slice columns
    auto col_slice = t.slice(1, 0, 2);
    EXPECT_EQ(col_slice.shape(), Shape({3, 2}));
    EXPECT_FLOAT_EQ((col_slice[{0, 0}]), 1.0F);
    EXPECT_FLOAT_EQ((col_slice[{0, 1}]), 2.0F);
    EXPECT_FLOAT_EQ((col_slice[{2, 1}]), 8.0F);
}

TEST_F(TensorTest, SlicingZeroCopy) {
    Tensor<float> t({1.0F, 2.0F, 3.0F, 4.0F, 5.0F});

    auto sliced = t.slice(0, 1, 4);

    // Modifying slice should affect original
    sliced[{0}] = 100.0F;
    EXPECT_FLOAT_EQ((t[{1}]), 100.0F);
}

TEST_F(TensorTest, SlicingOutOfBounds) {
    const Tensor<float> kT({1.0F, 2.0F, 3.0F, 4.0F, 5.0F});

    EXPECT_THROW(kT.slice(1, 0, 2), std::out_of_range);   // Dimension out of range
    EXPECT_THROW(kT.slice(0, 3, 2), std::out_of_range);   // start >= end
    EXPECT_THROW(kT.slice(0, 0, 10), std::out_of_range);  // end > size
}

// ========================================
// View Zero-Copy Tests
// ========================================

TEST_F(TensorTest, ViewZeroCopy) {
    Tensor<float> t({1.0F, 2.0F, 3.0F, 4.0F});

    auto viewed = t.view(Shape({2, 2}));

    // Verify data is shared (zero-copy)
    EXPECT_EQ(t.data(), viewed.data());

    // Modifying view affects original
    viewed[{0, 1}] = 100.0F;
    EXPECT_FLOAT_EQ((t[{1}]), 100.0F);

    // Modifying original affects view
    t[{3}] = 200.0F;
    EXPECT_FLOAT_EQ((viewed[{1, 1}]), 200.0F);
}

TEST_F(TensorTest, TransposeZeroCopyVerification) {
    Tensor<float> t({{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}});

    auto transposed = t.transpose(0, 1);

    // Verify data pointer is the same (zero-copy)
    EXPECT_EQ(t.data(), transposed.data());
}

TEST_F(TensorTest, SliceZeroCopyVerification) {
    Tensor<float> t({1.0F, 2.0F, 3.0F, 4.0F, 5.0F});

    auto sliced = t.slice(0, 1, 4);

    // Verify data is shared (may have different pointer due to offset)
    // But modifying one affects the other
    sliced[{0}] = 999.0F;
    EXPECT_FLOAT_EQ((t[{1}]), 999.0F);
}

// ========================================
// Contiguous Tests
// ========================================

TEST_F(TensorTest, ContiguousCheck) {
    // Newly created tensor is contiguous
    const Tensor<float> kT1({1.0F, 2.0F, 3.0F, 4.0F});
    EXPECT_TRUE(kT1.isContiguous());

    // View is contiguous
    auto viewed = kT1.view(Shape({2, 2}));
    EXPECT_TRUE(viewed.isContiguous());

    // Transposed tensor is not contiguous
    Tensor<float> t2({{1.0F, 2.0F}, {3.0F, 4.0F}});
    auto transposed = t2.transpose(0, 1);
    EXPECT_FALSE(transposed.isContiguous());

    // Sliced tensor may not be contiguous
    Tensor<float> t3({{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}});
    auto col_slice = t3.slice(1, 0, 2);
    EXPECT_FALSE(col_slice.isContiguous());
}

TEST_F(TensorTest, ContiguousOperation) {
    Tensor<float> t({{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}});
    auto transposed = t.transpose(0, 1);
    EXPECT_FALSE(transposed.isContiguous());

    // Make it contiguous
    auto contiguous = transposed.contiguous();

    EXPECT_TRUE(contiguous.isContiguous());
    EXPECT_EQ(contiguous.shape(), Shape({3, 2}));
    EXPECT_FLOAT_EQ((contiguous[{0, 0}]), 1.0F);
    EXPECT_FLOAT_EQ((contiguous[{0, 1}]), 4.0F);
    EXPECT_FLOAT_EQ((contiguous[{2, 1}]), 6.0F);

    // Original should not be affected
    EXPECT_FALSE(transposed.isContiguous());
}

TEST_F(TensorTest, ContiguousAlreadyContiguous) {
    const Tensor<float> kT({1.0F, 2.0F, 3.0F, 4.0F});

    EXPECT_TRUE(kT.isContiguous());

    auto contiguous = kT.contiguous();

    // Should return a copy
    EXPECT_TRUE(contiguous.isContiguous());
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
            EXPECT_FLOAT_EQ((t[{i, j}]), 0.0F);
        }
    }
}

TEST_F(TensorTest, FactoryOnes) {
    auto t = Tensor<float>::ones(Shape({2, 3}));

    EXPECT_EQ(t.shape(), Shape({2, 3}));
    EXPECT_EQ(t.size(), 6);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ((t[{i, j}]), 1.0F);
        }
    }
}

TEST_F(TensorTest, FactoryRandn) {
    auto t = Tensor<float>::randn(Shape({100}));

    EXPECT_EQ(t.shape(), Shape({100}));
    EXPECT_EQ(t.size(), 100);

    // Check that values are not all the same (probabilistic test)
    const float kFirstValue = t[{0}];
    bool has_different_value = false;
    for (size_t i = 1; i < 100; ++i) {
        if (std::abs(t[{i}] - kFirstValue) > 0.001F) {
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
        EXPECT_GE(t[{i}], 0.0F);
        EXPECT_LT(t[{i}], 1.0F);
    }
}

TEST_F(TensorTest, FactoryEye) {
    auto t = Tensor<float>::eye(3);

    EXPECT_EQ(t.shape(), Shape({3, 3}));
    EXPECT_EQ(t.size(), 9);

    // Check diagonal is 1
    EXPECT_FLOAT_EQ((t[{0, 0}]), 1.0F);
    EXPECT_FLOAT_EQ((t[{1, 1}]), 1.0F);
    EXPECT_FLOAT_EQ((t[{2, 2}]), 1.0F);

    // Check off-diagonal is 0
    EXPECT_FLOAT_EQ((t[{0, 1}]), 0.0F);
    EXPECT_FLOAT_EQ((t[{0, 2}]), 0.0F);
    EXPECT_FLOAT_EQ((t[{1, 0}]), 0.0F);
    EXPECT_FLOAT_EQ((t[{1, 2}]), 0.0F);
    EXPECT_FLOAT_EQ((t[{2, 0}]), 0.0F);
    EXPECT_FLOAT_EQ((t[{2, 1}]), 0.0F);
}

TEST_F(TensorTest, FactoryZerosLike) {
    Tensor<float> t({{1.0F, 2.0F}, {3.0F, 4.0F}});

    auto zeros = Tensor<float>::zerosLike(t);

    EXPECT_EQ(zeros.shape(), t.shape());
    EXPECT_EQ(zeros.device(), t.device());
    EXPECT_FLOAT_EQ((zeros[{0, 0}]), 0.0F);
    EXPECT_FLOAT_EQ((zeros[{1, 1}]), 0.0F);
}

TEST_F(TensorTest, FactoryOnesLike) {
    Tensor<float> t({{1.0F, 2.0F}, {3.0F, 4.0F}});

    auto ones = Tensor<float>::onesLike(t);

    EXPECT_EQ(ones.shape(), t.shape());
    EXPECT_EQ(ones.device(), t.device());
    EXPECT_FLOAT_EQ((ones[{0, 0}]), 1.0F);
    EXPECT_FLOAT_EQ((ones[{1, 1}]), 1.0F);
}
