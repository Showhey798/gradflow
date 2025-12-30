#include <gradflow/autograd/shape.hpp>
#include <gtest/gtest.h>

namespace gradflow {
namespace test {

// ========================================
// Shape Construction Tests
// ========================================

TEST(ShapeTest, Construction) {
    // Test default constructor
    Shape shape1;
    EXPECT_EQ(shape1.ndim(), 0);
    EXPECT_EQ(shape1.size(), 1);

    // Test construction from initializer list
    Shape shape2({2, 3, 4});
    EXPECT_EQ(shape2.ndim(), 3);
    EXPECT_EQ(shape2[0], 2);
    EXPECT_EQ(shape2[1], 3);
    EXPECT_EQ(shape2[2], 4);
    EXPECT_EQ(shape2.size(), 24);

    // Test construction from vector
    std::vector<size_t> dims = {5, 6};
    Shape shape3(dims);
    EXPECT_EQ(shape3.ndim(), 2);
    EXPECT_EQ(shape3[0], 5);
    EXPECT_EQ(shape3[1], 6);
    EXPECT_EQ(shape3.size(), 30);
}

TEST(ShapeTest, ElementAccess) {
    Shape shape({2, 3, 4});

    // Test read access
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);

    // Test at() with bounds checking
    EXPECT_EQ(shape.at(0), 2);
    EXPECT_EQ(shape.at(1), 3);
    EXPECT_EQ(shape.at(2), 4);

    // Test out of bounds
    EXPECT_THROW(shape.at(3), std::out_of_range);
}

TEST(ShapeTest, Equality) {
    Shape shape1({2, 3, 4});
    Shape shape2({2, 3, 4});
    Shape shape3({2, 3, 5});
    Shape shape4({2, 3});

    EXPECT_EQ(shape1, shape2);
    EXPECT_NE(shape1, shape3);
    EXPECT_NE(shape1, shape4);
}

// ========================================
// Broadcasting Tests
// ========================================

TEST(ShapeTest, BroadcastCompatibility) {
    // Compatible shapes
    Shape shape1({3, 1, 4});
    Shape shape2({1, 2, 4});
    EXPECT_TRUE(shape1.is_broadcastable_with(shape2));

    // Same shape is always compatible
    Shape shape3({2, 3, 4});
    Shape shape4({2, 3, 4});
    EXPECT_TRUE(shape3.is_broadcastable_with(shape4));

    // Scalar (empty shape) is compatible with any shape
    Shape scalar;
    Shape shape5({2, 3, 4});
    EXPECT_TRUE(scalar.is_broadcastable_with(shape5));
    EXPECT_TRUE(shape5.is_broadcastable_with(scalar));

    // Incompatible shapes
    Shape shape6({3, 2, 4});
    Shape shape7({2, 3, 4});
    EXPECT_FALSE(shape6.is_broadcastable_with(shape7));
}

TEST(ShapeTest, BroadcastShape) {
    // Test broadcasting result shape
    Shape shape1({3, 1, 4});
    Shape shape2({1, 2, 4});
    Shape result = shape1.broadcast_with(shape2);
    EXPECT_EQ(result, Shape({3, 2, 4}));

    // Broadcasting with scalar
    Shape scalar;
    Shape shape3({2, 3, 4});
    Shape result2 = scalar.broadcast_with(shape3);
    EXPECT_EQ(result2, shape3);

    // Broadcasting with different ndim
    Shape shape4({4});
    Shape shape5({2, 3, 4});
    Shape result3 = shape4.broadcast_with(shape5);
    EXPECT_EQ(result3, Shape({2, 3, 4}));
}

TEST(ShapeTest, BroadcastIncompatible) {
    Shape shape1({3, 2, 4});
    Shape shape2({2, 3, 4});
    EXPECT_THROW(shape1.broadcast_with(shape2), std::invalid_argument);
}

// ========================================
// Stride Tests
// ========================================

TEST(StrideTest, RowMajorStride) {
    // Test 1D
    Stride stride1(Shape({5}));
    EXPECT_EQ(stride1.ndim(), 1);
    EXPECT_EQ(stride1[0], 1);

    // Test 2D (row-major: [n, m] -> strides [m, 1])
    Stride stride2(Shape({3, 4}));
    EXPECT_EQ(stride2.ndim(), 2);
    EXPECT_EQ(stride2[0], 4);
    EXPECT_EQ(stride2[1], 1);

    // Test 3D (row-major: [a, b, c] -> strides [b*c, c, 1])
    Stride stride3(Shape({2, 3, 4}));
    EXPECT_EQ(stride3.ndim(), 3);
    EXPECT_EQ(stride3[0], 12);  // 3 * 4
    EXPECT_EQ(stride3[1], 4);
    EXPECT_EQ(stride3[2], 1);
}

TEST(StrideTest, OffsetCalculation) {
    // Test 2D offset calculation
    Stride stride2d(Shape({3, 4}));
    EXPECT_EQ(stride2d.offset({0, 0}), 0);
    EXPECT_EQ(stride2d.offset({0, 1}), 1);
    EXPECT_EQ(stride2d.offset({1, 0}), 4);
    EXPECT_EQ(stride2d.offset({1, 1}), 5);
    EXPECT_EQ(stride2d.offset({2, 3}), 11);

    // Test 3D offset calculation
    Stride stride3d(Shape({2, 3, 4}));
    EXPECT_EQ(stride3d.offset({0, 0, 0}), 0);
    EXPECT_EQ(stride3d.offset({0, 0, 1}), 1);
    EXPECT_EQ(stride3d.offset({0, 1, 0}), 4);
    EXPECT_EQ(stride3d.offset({1, 0, 0}), 12);
    EXPECT_EQ(stride3d.offset({1, 2, 3}), 23);  // 1*12 + 2*4 + 3*1
}

TEST(StrideTest, CustomStride) {
    // Test custom stride (e.g., for transposed view)
    std::vector<size_t> custom_strides = {1, 3};
    Stride stride(custom_strides);
    EXPECT_EQ(stride.ndim(), 2);
    EXPECT_EQ(stride[0], 1);
    EXPECT_EQ(stride[1], 3);

    // Offset with custom stride
    EXPECT_EQ(stride.offset({0, 0}), 0);
    EXPECT_EQ(stride.offset({1, 0}), 1);
    EXPECT_EQ(stride.offset({0, 1}), 3);
    EXPECT_EQ(stride.offset({2, 1}), 5);  // 2*1 + 1*3
}

TEST(StrideTest, IsContiguous) {
    // Row-major strides are contiguous
    Stride stride1(Shape({3, 4}));
    EXPECT_TRUE(stride1.isContiguous(Shape({3, 4})));

    // Custom strides may not be contiguous
    Stride stride2(std::vector<size_t>({1, 3}));
    EXPECT_FALSE(stride2.isContiguous(Shape({3, 4})));

    // Transposed view is not contiguous
    Stride stride3(std::vector<size_t>({1, 4}));
    EXPECT_FALSE(stride3.isContiguous(Shape({4, 3})));
}

}  // namespace test
}  // namespace gradflow
