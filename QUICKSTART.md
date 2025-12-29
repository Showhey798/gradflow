# Quick Start Guide

Get up and running with FullScratchML in 5 minutes!

## Prerequisites

- C++17 compatible compiler (GCC 11+, Clang 15+, MSVC 193+)
- CMake 3.20+
- Python 3.8+ (for Conan package manager)

## Installation

### Option 1: Using the Build Script (Recommended)

```bash
# Install Conan (if not already installed)
pip install conan==2.0.17

# Run the build script
./scripts/build.sh
```

That's it! The script will:
1. Install dependencies via Conan
2. Configure CMake
3. Build the project
4. Run tests

### Option 2: Manual Build

```bash
# Install dependencies
pip install conan==2.0.17
conan profile detect --force
conan install . --build=missing

# Configure and build
cmake --preset conan-release
cmake --build --preset conan-release

# Run tests
ctest --preset conan-release
```

## Run Your First Example

```bash
# After building, run the matrix operations example
./build/Release/examples/matrix_operations
```

You should see output like:
```
=== FullScratchML Matrix Operations Example ===

Matrix A (2x3):
1 2 3
4 5 6

Matrix B (3x2):
7 8
9 10
11 12

Matrix C = A * B (2x2):
58 64
139 154

...
```

## Next Steps

### 1. Explore Examples

Check out the examples in the `examples/` directory:
- `matrix_operations.cpp`: Basic matrix operations
- `simple_neural_network.cpp`: Neural network skeleton

### 2. Write Your First Code

Create a new file `my_first_ml.cpp`:

```cpp
#include <fullscratch/fullscratch.hpp>
#include <iostream>

int main() {
    // Create a 2x2 matrix
    fullscratch::Matrix<double> a = {{1.0, 2.0}, {3.0, 4.0}};

    // Multiply by 2
    auto b = a * 2.0;

    // Print result
    for (size_t i = 0; i < b.rows(); ++i) {
        for (size_t j = 0; j < b.cols(); ++j) {
            std::cout << b(i, j) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
```

Compile and run:
```bash
g++ -std=c++17 -I include my_first_ml.cpp -o my_first_ml
./my_first_ml
```

### 3. Learn More

- Read the [Build Instructions](docs/BUILD.md) for advanced build options
- Check [Contributing Guidelines](docs/CONTRIBUTING.md) to contribute
- Browse the [API Documentation](include/fullscratch/) for available features

## Troubleshooting

### Conan not found

```bash
pip install conan==2.0.17
```

### Compiler not found

Make sure you have a C++17 compatible compiler installed:

**Ubuntu/Debian:**
```bash
sudo apt-get install build-essential
```

**macOS:**
```bash
xcode-select --install
```

**Windows:**
Install Visual Studio 2022 with C++ workload

### Build fails

Try a clean build:
```bash
./scripts/build.sh --clean
```

## Getting Help

- Open an issue on GitHub
- Check the documentation in `docs/`
- Read the FAQ (coming soon)

Happy coding!
