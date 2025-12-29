# Contributing to FullScratchML

Thank you for your interest in contributing to FullScratchML! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, constructive, and collaborative. We aim to create an inclusive environment for all contributors.

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/fullScratchLibs.git
cd fullScratchLibs
```

### 2. Set Up Development Environment

```bash
# Install dependencies
conan install . --build=missing

# Configure for development (with tests and examples)
cmake --preset conan-debug \
    -DFULLSCRATCH_BUILD_TESTS=ON \
    -DFULLSCRATCH_BUILD_EXAMPLES=ON \
    -DFULLSCRATCH_ENABLE_COVERAGE=ON

# Build
cmake --build --preset conan-debug
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## Development Workflow

### Test-Driven Development (TDD)

We follow the TDD cycle:

1. **Red**: Write a failing test
2. **Green**: Write minimal code to make it pass
3. **Refactor**: Improve the code while keeping tests passing

Example:
```cpp
// 1. Red - Write failing test in tests/test_new_feature.cpp
TEST(NewFeatureTest, BasicFunctionality) {
    auto result = new_feature(input);
    EXPECT_EQ(result, expected);
}

// 2. Green - Implement in include/fullscratch/new_feature.hpp
// (minimal implementation)

// 3. Refactor - Improve code quality while tests pass
```

### Code Quality Standards

#### C++ Standards

- **C++17**: Use modern C++ features
- **Header-only preferred**: When possible
- **RAII**: Resource management via constructors/destructors
- **const-correctness**: Mark functions and parameters const when appropriate
- **No raw pointers**: Use smart pointers or references

#### Naming Conventions

```cpp
namespace fullscratch {

class MyClass {          // PascalCase for classes
  public:
    void my_method();    // snake_case for methods

  private:
    int member_;         // trailing underscore for private members
};

constexpr int MAX_SIZE = 100;  // UPPER_CASE for constants

template <typename T>  // PascalCase for template parameters
void my_function(int parameter_name);  // snake_case for parameters

}  // namespace fullscratch
```

### Code Formatting

Before committing, format your code:

```bash
# Format all modified files
git diff --name-only | grep -E '\.(cpp|hpp|h)$' | xargs clang-format -i

# Or format specific files
clang-format -i include/fullscratch/my_file.hpp
```

### Static Analysis

Run clang-tidy on your changes:

```bash
# Build with compile commands
cmake --preset conan-debug

# Run clang-tidy on specific files
clang-tidy -p build/Debug include/fullscratch/my_file.hpp
```

## Testing Guidelines

### Writing Tests

1. **Test file naming**: `test_<module>.cpp`
2. **Test naming**: `TEST(ModuleTest, DescriptiveTestName)`
3. **One assertion per test**: Focus each test on a single behavior

Example:
```cpp
#include <gtest/gtest.h>
#include <fullscratch/matrix.hpp>

namespace fullscratch {
namespace test {

class MatrixTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Set up test fixtures
        std::srand(42);  // Fixed seed for reproducibility
    }
};

TEST_F(MatrixTest, AdditionIsCommutative) {
    Matrix<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    Matrix<double> b = {{5.0, 6.0}, {7.0, 8.0}};

    auto ab = a + b;
    auto ba = b + a;

    EXPECT_EQ(ab, ba);
}

}  // namespace test
}  // namespace fullscratch
```

### Running Tests

```bash
# Run all tests
cmake --build --preset conan-debug --target check

# Run specific test
./build/Debug/tests/matrix_test

# Run with filters
./build/Debug/tests/matrix_test --gtest_filter=MatrixTest.Addition*

# Run with verbose output
./build/Debug/tests/matrix_test --gtest_verbose
```

### Test Coverage

Aim for high test coverage:

```bash
# Build with coverage
cmake --preset conan-debug -DFULLSCRATCH_ENABLE_COVERAGE=ON
cmake --build --preset conan-debug

# Run tests
ctest --preset conan-debug

# Generate coverage report
lcov --directory . --capture --output-file coverage.info
lcov --list coverage.info
```

## Documentation

### Code Documentation

Use Doxygen-style comments:

```cpp
/**
 * @brief Brief description of the function
 *
 * Detailed description with more information about the function's
 * behavior, parameters, and return value.
 *
 * @param input Description of input parameter
 * @param output Description of output parameter
 * @return Description of return value
 *
 * @throws std::invalid_argument If input is invalid
 *
 * @example
 * @code
 * auto result = my_function(42, output);
 * @endcode
 */
int my_function(int input, int& output);
```

### README Updates

If adding a new feature, update:
- `README.md`: Feature list and examples
- `docs/BUILD.md`: Build instructions if needed
- Example code in `examples/`

## Commit Guidelines

### Commit Messages

Follow the conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Build process or auxiliary tool changes

Example:
```
feat(matrix): add element-wise multiplication

Implement element-wise (Hadamard) product for matrices.
This is useful for certain neural network operations.

Closes #42
```

### Commit Frequency

- Commit early and often
- Each commit should be logical and self-contained
- All tests must pass before committing
- No broken commits in history

## Pull Request Process

### Before Submitting

1. **Rebase on main**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   # Format code
   find include src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i

   # Build and test
   cmake --build --preset conan-debug
   ctest --preset conan-debug --output-on-failure

   # Run static analysis
   clang-tidy -p build/Debug $(git diff --name-only main | grep -E '\.(cpp|hpp)$')
   ```

3. **Update documentation**

### Submitting PR

1. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create PR on GitHub

3. Fill out the PR template:
   - Description of changes
   - Related issues
   - Testing performed
   - Breaking changes (if any)

### PR Review Process

- Maintainers will review within 1-2 weeks
- Address feedback with new commits
- Once approved, maintainer will merge

### After Merge

Delete your branch:
```bash
git checkout main
git pull upstream main
git branch -d feature/your-feature-name
```

## Architecture Guidelines

### ML-Specific Considerations

#### Reproducibility

```cpp
// Always set random seeds in tests
TEST(ReproducibilityTest, FixedSeed) {
    std::srand(42);
    // test code
}

// Document non-determinism
/**
 * @note This function uses random initialization.
 *       Set random seed before calling for reproducibility.
 */
void random_initialization();
```

#### Data Integrity

```cpp
// Never modify raw input data
const Matrix<T>& get_data() const;  // Return const reference

// Keep train/val/test separate
class DataLoader {
    Dataset train_data_;  // Never touch test data here
    Dataset val_data_;
    Dataset test_data_;
};
```

#### Configuration Management

```cpp
// Use config structs instead of magic numbers
struct TrainingConfig {
    double learning_rate = 0.001;
    int batch_size = 32;
    int epochs = 100;
};

void train(const TrainingConfig& config);
```

### Performance Considerations

```cpp
// Use move semantics
Matrix<T> multiply(Matrix<T>&& a, Matrix<T>&& b);

// Avoid unnecessary copies
const Matrix<T>& get_weights() const;  // Return reference

// Vectorize operations
// Prefer Eigen or SIMD over loops when possible
```

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue with reproducible example
- **Chat**: Join our community channel (link in README)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
