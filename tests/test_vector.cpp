#include <gtest/gtest.h>

#include <gradflow/vector.hpp>

namespace gradflow {
namespace test {

class VectorTest : public ::testing::Test {
  protected:
    void SetUp() override { std::srand(42); }
};

TEST_F(VectorTest, Placeholder) {
    // TODO: Implement vector tests
    EXPECT_TRUE(true);
}

}  // namespace test
}  // namespace gradflow
