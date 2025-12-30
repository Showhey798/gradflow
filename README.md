# GradFlow

A comprehensive C++ machine learning library featuring automatic differentiation and Transformer implementation, with first-class support for Apple Silicon Metal GPU acceleration.

## Features

- **Automatic Differentiation**: Both dynamic (Define-by-Run) and static (Define-and-Run) computational graphs
- **Transformer Architecture**: Full implementation of multi-head attention and encoder-decoder
- **Multi-Backend GPU Support**:
  - **Metal**: Optimized for Apple Silicon (M1/M2/M3) with Metal Performance Shaders
  - **CUDA**: NVIDIA GPU support with cuBLAS integration (optional)
- **Modern C++17**: Leveraging latest C++ standards with SOLID principles
- **Python Bindings**: PyTorch-like Python interface via nanobind (high-performance, compact)
- **Comprehensive Testing**: Google Test framework with numerical gradient checking
- **Production Ready**: Memory pooling, kernel fusion, and SIMD optimizations

## Planned Modules

### Core Infrastructure (Phase 1-2)
- **Tensor Operations**: N-dimensional arrays with striding and broadcasting
- **Autograd Engine**: Dynamic computational graphs with automatic differentiation
- **Optimizers**: SGD, Adam, AdamW with learning rate schedulers

### Metal GPU Acceleration (Phase 3) - Priority
- **Metal Backend**: High-performance GPU kernels with Metal Performance Shaders (MPS)
- **Unified Memory**: Efficient CPU-GPU data transfer on Apple Silicon
- **Neural Engine**: Leverage Apple's dedicated ML hardware

### Neural Network Components (Phase 4-5)
- **Activation Functions**: ReLU, GELU, Sigmoid, Tanh, Softmax
- **Layer Normalization**: Stable training for deep networks
- **Attention Mechanisms**: Scaled dot-product and multi-head attention
- **Transformer Layers**: Complete encoder-decoder architecture

### Advanced Features (Phase 7-8)
- **Static Graph**: Define-and-Run computational graphs with global optimizations
- **CUDA Support**: Optional NVIDIA GPU backend for broader hardware support

## Documentation

Comprehensive design documents are available in the `docs/` directory:

- **[Library Naming](docs/LIBRARY_NAMING.md)**: Rationale for the GradFlow name
- **[Architecture Design](docs/ARCHITECTURE.md)**: System architecture, core components, and design patterns
- **[API Design](docs/API_DESIGN.md)**: Complete API reference with C++ and Python examples
- **[Implementation Roadmap](docs/ROADMAP.md)**: Phased implementation plan with milestones
- **[Technical Decisions](docs/TECHNICAL_DECISIONS.md)**: Detailed rationale for key technical choices

## Requirements

### Build Tools
- CMake >= 3.20
- C++17 compatible compiler:
  - GCC >= 11
  - Clang >= 15
  - MSVC >= 193 (Visual Studio 2022)
- Conan >= 2.0 (for dependency management)
- **macOS 12.0+** with Xcode Command Line Tools (for Metal support)

### Dependencies
- Eigen3 >= 3.4 (linear algebra)
- Google Test >= 1.14 (testing)
- nanobind >= 2.0 (Python bindings, optional)
- Metal Performance Shaders (macOS only)

## Quick Start

### Installation

#### Using Conan (Recommended)

```bash
# Install dependencies
conan install . --build=missing

# Configure
cmake --preset conan-release

# Build
cmake --build --preset conan-release

# Run tests
ctest --preset conan-release
```

#### Manual Build

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --parallel

# Run tests
ctest --output-on-failure
```

### Usage Example

```cpp
#include <gradflow/gradflow.hpp>

int main() {
    // Create tensors on Metal GPU (Apple Silicon)
    auto a = gradflow::Tensor<float>::randn({100, 200}, gradflow::DeviceType::Metal);
    auto b = gradflow::Tensor<float>::randn({200, 50}, gradflow::DeviceType::Metal);

    // Matrix operations with automatic differentiation
    auto c = gradflow::matmul(a, b);
    auto d = gradflow::relu(c);

    // Backward pass
    auto loss = d.sum();
    loss.backward();

    return 0;
}
```

## Build Options

- `GRADFLOW_BUILD_TESTS`: Build tests (default: ON)
- `GRADFLOW_BUILD_EXAMPLES`: Build examples (default: ON)
- `GRADFLOW_BUILD_BENCHMARKS`: Build benchmarks (default: OFF)
- `GRADFLOW_BUILD_PYTHON`: Build Python bindings (default: OFF)
- `GRADFLOW_BUILD_METAL`: Build Metal backend (default: ON on macOS)
- `GRADFLOW_BUILD_CUDA`: Build CUDA backend (default: OFF)
- `GRADFLOW_ENABLE_COVERAGE`: Enable code coverage (default: OFF)
- `GRADFLOW_ENABLE_SANITIZERS`: Enable sanitizers (default: OFF)

Example:
```bash
cmake -B build -DGRADFLOW_BUILD_PYTHON=ON -DGRADFLOW_BUILD_METAL=ON
```

## Development

### CI Verification (Recommended)

Run all CI checks locally before pushing:

```bash
# Make scripts executable (first time only)
chmod +x scripts/*.sh

# Run all CI checks (format, lint, tests)
bash scripts/ci-verify.sh

# Or run individual checks:
bash scripts/ci-format-check.sh    # C++ format check
bash scripts/ci-lint-python.sh     # Python linting
bash scripts/ci-test-cpp.sh        # C++ tests
bash scripts/ci-test-python.sh     # Python tests
```

### Code Formatting

```bash
# Format all source files
find include src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i

# Or use the script
bash scripts/ci-format-apply.sh
```

### Static Analysis

```bash
# Run clang-tidy
cmake --build build --target clang-tidy
```

### Running Tests

```bash
# Run all tests
cmake --build build --target check

# Run specific test
./build/tests/tensor_test

# Run Python tests (if bindings are built)
pytest python/tests -v
```

### Code Coverage

```bash
# Configure with coverage enabled
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DGRADFLOW_ENABLE_COVERAGE=ON

# Build and run tests
cmake --build build
ctest --test-dir build

# Generate coverage report
lcov --directory build --capture --output-file coverage.info
lcov --remove coverage.info '/usr/*' '*/tests/*' --output-file coverage.info
genhtml coverage.info --output-directory coverage_report
```

## Python Bindings

```bash
# Build Python bindings
cmake -B build -DGRADFLOW_BUILD_PYTHON=ON
cmake --build build

# Install Python package
pip install ./python
```

Usage:
```python
import gradflow as gf
import gradflow.nn as nn

# Create tensors
x = gf.randn((32, 128), device='metal')  # Apple Silicon GPU
w = gf.randn((128, 10), device='metal', requires_grad=True)

# Forward pass
y = gf.matmul(x, w)
y = gf.relu(y)

# Loss and backward
loss = y.sum()
loss.backward()

# Access gradients
grad_w = w.grad
```

## Project Structure

```
GradFlow/
├── include/gradflow/       # Public header files
│   ├── autograd/          # Automatic differentiation
│   ├── nn/                # Neural network components
│   └── optim/             # Optimizers
├── src/                   # Source files
│   ├── autograd/
│   │   ├── cpu/          # CPU implementations
│   │   ├── metal/        # Metal GPU implementations
│   │   └── cuda/         # CUDA implementations (optional)
│   └── nn/
├── tests/                 # Unit and integration tests
├── examples/              # Usage examples
├── python/                # Python bindings (nanobind)
├── benchmarks/            # Performance benchmarks
├── docs/                  # Documentation
├── cmake/                 # CMake modules
├── scripts/               # Build and utility scripts
├── .github/workflows/     # CI/CD configurations
├── CMakeLists.txt         # Main CMake configuration
├── conanfile.py           # Conan package file
└── README.md              # This file
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow the existing code style (enforced by clang-format)
- Write comprehensive tests for new features
- Document public APIs with Doxygen comments
- Ensure all CI checks pass

## License

MIT License - see LICENSE file for details

## Acknowledgments

- PyTorch for API design inspiration
- Eigen library for linear algebra
- Apple Metal Performance Shaders for GPU acceleration
- Google Test for testing framework
- nanobind for efficient Python bindings
