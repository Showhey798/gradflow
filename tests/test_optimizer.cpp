#include <cmath>

#include <gradflow/autograd/tensor.hpp>
#include <gradflow/autograd/variable.hpp>
#include <gradflow/optim/adam.hpp>
#include <gradflow/optim/optimizer.hpp>
#include <gradflow/optim/sgd.hpp>
#include <gtest/gtest.h>

namespace gradflow {
namespace test {

/**
 * @brief Test fixture for optimizer tests
 */
class OptimizerTest : public ::testing::Test {
protected:
    /**
     * @brief Helper function to optimize a quadratic function: f(x) = (x - target)^2
     *
     * @param optimizer The optimizer to use
     * @param param The parameter to optimize
     * @param target The target value (minimum of the quadratic)
     * @param num_steps Number of optimization steps
     * @return The final parameter value
     */
    template <typename T>
    T optimize_quadratic(optim::Optimizer<T>* optimizer,
                         Variable<T>& param,
                         T target,
                         int num_steps) {
        // Initialize gradient once
        param.zeroGrad();

        for (int step = 0; step < num_steps; ++step) {
            // Compute loss: f(x) = (x - target)^2
            T x = param.data().data()[0];

            // Compute gradient: df/dx = 2 * (x - target)
            T grad_value = 2 * (x - target);

            // Set gradient manually (since we're using a simple scalar function)
            param.grad().data()[0] = grad_value;

            // Update parameters
            optimizer->step();

            // Zero gradients for next iteration
            optimizer->zeroGrad();
        }

        return param.data().data()[0];
    }
};

/**
 * @brief Test SGD convergence on a simple quadratic function
 */
TEST_F(OptimizerTest, SGDConvergence) {
    // Test problem: minimize f(x) = (x - 3)^2
    // Starting from x = 0, should converge to x = 3

    float initial_value = 0.0f;
    float target_value = 3.0f;
    float learning_rate = 0.1f;
    int num_steps = 100;

    // Create parameter (1D tensor with 1 element)
    Tensor<float> param_data(Shape{1});
    param_data.data()[0] = initial_value;
    auto param = Variable<float>(param_data, true);

    // Create SGD optimizer
    auto optimizer = optim::SGD<float>(learning_rate);
    optimizer.addParamGroup({&param});

    // Optimize
    float final_value = optimize_quadratic(&optimizer, param, target_value, num_steps);

    // Check convergence (should converge within 2% of target)
    EXPECT_NEAR(final_value, target_value, 0.06);  // 2% of 3.0
}

/**
 * @brief Test SGD with momentum
 */
TEST_F(OptimizerTest, SGDWithMomentum) {
    // Test problem: minimize f(x) = (x - 5)^2
    // With momentum, should converge faster

    float initial_value = 0.0f;
    float target_value = 5.0f;
    float learning_rate = 0.05f;
    float momentum = 0.9f;
    int num_steps = 100;

    // Create parameter (scalar tensor)
    Tensor<float> param_data(Shape{1});
    param_data.data()[0] = initial_value;
    auto param = Variable<float>(param_data, true);

    // Create SGD optimizer with momentum
    auto optimizer = optim::SGD<float>(learning_rate, momentum);
    optimizer.addParamGroup({&param});

    // Optimize
    float final_value = optimize_quadratic(&optimizer, param, target_value, num_steps);

    // Check convergence (momentum should help convergence, within 2% of target)
    EXPECT_NEAR(final_value, target_value, 0.10);  // 2% of 5.0
}

/**
 * @brief Test Adam convergence on a simple quadratic function
 */
TEST_F(OptimizerTest, AdamConvergence) {
    // Test problem: minimize f(x) = (x - 3)^2
    // Starting from x = 0, should converge to x = 3

    float initial_value = 0.0f;
    float target_value = 3.0f;
    int num_steps = 100;

    // Create parameter (scalar tensor)
    Tensor<float> param_data(Shape{1});
    param_data.data()[0] = initial_value;
    auto param = Variable<float>(param_data, true);

    // Create Adam optimizer with higher learning rate for faster convergence
    float learning_rate = 0.1f;
    auto optimizer = optim::Adam<float>(learning_rate);
    optimizer.addParamGroup({&param});

    // Optimize
    float final_value = optimize_quadratic(&optimizer, param, target_value, num_steps);

    // Check convergence (Adam should converge well, within 2% of target)
    EXPECT_NEAR(final_value, target_value, 0.06);  // 2% of 3.0
}

/**
 * @brief Test AdamW (Adam with weight decay)
 */
TEST_F(OptimizerTest, AdamWConvergence) {
    // Test problem: minimize f(x) = (x - 3)^2 with weight decay
    // AdamW applies weight decay directly to parameters

    float initial_value = 0.0f;
    float target_value = 3.0f;
    float weight_decay = 0.01f;
    int num_steps = 100;

    // Create parameter (scalar tensor)
    Tensor<float> param_data(Shape{1});
    param_data.data()[0] = initial_value;
    auto param = Variable<float>(param_data, true);

    // Create AdamW optimizer with higher learning rate
    float learning_rate = 0.1f;
    auto optimizer = optim::Adam<float>(learning_rate, 0.9f, 0.999f, 1e-8f, weight_decay, true);
    optimizer.addParamGroup({&param});

    // Optimize
    float final_value = optimize_quadratic(&optimizer, param, target_value, num_steps);

    // With weight decay, the convergence point should be slightly pulled toward 0
    // But should still converge close to target (within 2%)
    EXPECT_NEAR(final_value, target_value, 0.06);  // 2% of 3.0
}

/**
 * @brief Test zero_grad functionality
 */
TEST_F(OptimizerTest, ZeroGrad) {
    // Create parameter with a gradient
    Tensor<float> param_data(Shape{1});
    param_data.data()[0] = 1.0f;
    auto param = Variable<float>(param_data, true);

    // Set gradient
    param.zeroGrad();
    param.grad().data()[0] = 5.0f;

    // Create optimizer and add parameter
    auto optimizer = optim::SGD<float>(0.1f);
    optimizer.addParamGroup({&param});

    // Verify gradient exists
    EXPECT_TRUE(param.hasGrad());
    EXPECT_FLOAT_EQ(param.grad().data()[0], 5.0f);

    // Zero gradients
    optimizer.zeroGrad();

    // Verify gradient is zeroed
    EXPECT_TRUE(param.hasGrad());
    EXPECT_FLOAT_EQ(param.grad().data()[0], 0.0f);
}

/**
 * @brief Test optimizer with multiple parameters
 */
TEST_F(OptimizerTest, MultipleParameters) {
    // Create two parameters
    Tensor<float> param1_data(Shape{1});
    param1_data.data()[0] = 0.0f;
    auto param1 = Variable<float>(param1_data, true);

    Tensor<float> param2_data(Shape{1});
    param2_data.data()[0] = 0.0f;
    auto param2 = Variable<float>(param2_data, true);

    // Create optimizer
    auto optimizer = optim::SGD<float>(0.1f);
    optimizer.addParamGroup({&param1, &param2});

    // Verify number of parameters
    EXPECT_EQ(optimizer.num_params(), 2);

    // Set gradients
    param1.zeroGrad();
    param1.grad().data()[0] = 1.0f;
    param2.zeroGrad();
    param2.grad().data()[0] = 2.0f;

    // Perform optimization step
    optimizer.step();

    // Check that both parameters were updated
    EXPECT_FLOAT_EQ(param1.data().data()[0], -0.1f);  // 0.0 - 0.1 * 1.0
    EXPECT_FLOAT_EQ(param2.data().data()[0], -0.2f);  // 0.0 - 0.1 * 2.0
}

/**
 * @brief Test SGD weight decay
 */
TEST_F(OptimizerTest, SGDWeightDecay) {
    float initial_value = 10.0f;
    float learning_rate = 0.1f;
    float weight_decay = 0.1f;

    // Create parameter
    Tensor<float> param_data(Shape{1});
    param_data.data()[0] = initial_value;
    auto param = Variable<float>(param_data, true);

    // Set zero gradient
    param.zeroGrad();
    param.grad().data()[0] = 0.0f;

    // Create optimizer with weight decay
    auto optimizer = optim::SGD<float>(learning_rate, 0.0f, weight_decay);
    optimizer.addParamGroup({&param});

    // Perform one step
    optimizer.step();

    // With zero gradient, only weight decay should be applied
    // Expected: param = 10.0 - 0.1 * (0.0 + 0.1 * 10.0) = 10.0 - 0.1 = 9.9
    float expected = initial_value - learning_rate * weight_decay * initial_value;
    EXPECT_NEAR(param.data().data()[0], expected, 1e-5);
}

/**
 * @brief Test Adam step count
 */
TEST_F(OptimizerTest, AdamStepCount) {
    // Create parameter
    Tensor<float> param_data(Shape{1});
    param_data.data()[0] = 1.0f;
    auto param = Variable<float>(param_data, true);

    // Create Adam optimizer
    auto optimizer = optim::Adam<float>();
    optimizer.addParamGroup({&param});

    // Initial step count should be 0
    EXPECT_EQ(optimizer.stepCount(), 0);

    // Set gradient and perform steps
    for (int i = 1; i <= 5; ++i) {
        if (!param.hasGrad()) {
            param.zeroGrad();
        }
        param.grad().data()[0] = 1.0f;
        optimizer.step();
        EXPECT_EQ(optimizer.stepCount(), static_cast<size_t>(i));
    }
}

/**
 * @brief Test optimizer hyperparameter getters
 */
TEST_F(OptimizerTest, HyperparameterGetters) {
    // Test SGD
    auto sgd = optim::SGD<float>(0.01f, 0.9f, 0.001f);
    EXPECT_FLOAT_EQ(sgd.lr(), 0.01f);
    EXPECT_FLOAT_EQ(sgd.momentum(), 0.9f);
    EXPECT_FLOAT_EQ(sgd.weightDecay(), 0.001f);

    // Test Adam
    auto adam = optim::Adam<float>(0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, true);
    EXPECT_FLOAT_EQ(adam.lr(), 0.001f);
    EXPECT_FLOAT_EQ(adam.beta1(), 0.9f);
    EXPECT_FLOAT_EQ(adam.beta2(), 0.999f);
    EXPECT_FLOAT_EQ(adam.epsilon(), 1e-8f);
    EXPECT_FLOAT_EQ(adam.weightDecay(), 0.01f);
    EXPECT_TRUE(adam.isAdamw());
}

/**
 * @brief Test learning rate setter
 */
TEST_F(OptimizerTest, SetLearningRate) {
    // Test SGD
    auto sgd = optim::SGD<float>(0.01f);
    EXPECT_FLOAT_EQ(sgd.lr(), 0.01f);

    sgd.setLr(0.001f);
    EXPECT_FLOAT_EQ(sgd.lr(), 0.001f);

    // Test Adam
    auto adam = optim::Adam<float>(0.01f);
    EXPECT_FLOAT_EQ(adam.lr(), 0.01f);

    adam.setLr(0.001f);
    EXPECT_FLOAT_EQ(adam.lr(), 0.001f);
}

/**
 * @brief Test invalid hyperparameters throw exceptions
 */
TEST_F(OptimizerTest, InvalidHyperparameters) {
    // Invalid learning rate
    EXPECT_THROW(optim::SGD<float>(-0.01f), std::invalid_argument);
    EXPECT_THROW(optim::Adam<float>(-0.01f), std::invalid_argument);

    // Invalid momentum
    EXPECT_THROW(optim::SGD<float>(0.01f, -0.1f), std::invalid_argument);
    EXPECT_THROW(optim::SGD<float>(0.01f, 1.0f), std::invalid_argument);

    // Invalid beta1/beta2
    EXPECT_THROW(optim::Adam<float>(0.01f, -0.1f), std::invalid_argument);
    EXPECT_THROW(optim::Adam<float>(0.01f, 0.9f, 1.0f), std::invalid_argument);

    // Invalid epsilon
    EXPECT_THROW(optim::Adam<float>(0.01f, 0.9f, 0.999f, -1e-8f), std::invalid_argument);

    // Invalid weight decay
    EXPECT_THROW(optim::SGD<float>(0.01f, 0.0f, -0.01f), std::invalid_argument);
    EXPECT_THROW(optim::Adam<float>(0.01f, 0.9f, 0.999f, 1e-8f, -0.01f), std::invalid_argument);
}

}  // namespace test
}  // namespace gradflow
