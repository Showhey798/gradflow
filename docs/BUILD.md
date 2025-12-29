# Build Instructions

This document provides detailed instructions for building FullScratchML on different platforms.

## Prerequisites

### All Platforms

- CMake 3.20 or higher
- Python 3.8+ (for Conan)
- Git

### Platform-Specific

#### Linux

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    python3-pip \
    clang-15 \
    clang-format-15 \
    clang-tidy-15

# Install Conan
pip3 install conan==2.0.17
```

#### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake ninja python conan
```

#### Windows

1. Install Visual Studio 2022 (Community Edition or higher)
   - Include "Desktop development with C++" workload
2. Install CMake from https://cmake.org/download/
3. Install Python from https://www.python.org/downloads/
4. Install Conan:
   ```powershell
   pip install conan==2.0.17
   ```

## Building with Conan

### Step 1: Setup Conan Profile

```bash
# Detect default profile
conan profile detect --force

# (Optional) Customize profile
conan profile show default
```

### Step 2: Install Dependencies

```bash
# Install dependencies
conan install . --build=missing -s build_type=Release
```

### Step 3: Configure CMake

```bash
# Use Conan-generated preset
cmake --preset conan-release
```

### Step 4: Build

```bash
# Build the project
cmake --build --preset conan-release --parallel
```

### Step 5: Run Tests

```bash
# Run all tests
ctest --preset conan-release --output-on-failure
```

## Building Manually (Without Conan)

### Prerequisites

Install dependencies manually:

#### Linux

```bash
# Eigen3
sudo apt-get install libeigen3-dev

# Google Test
sudo apt-get install libgtest-dev
```

#### macOS

```bash
brew install eigen googletest
```

### Build Steps

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
cmake --build . --parallel $(nproc)

# Run tests
ctest --output-on-failure
```

## Build Variants

### Debug Build

```bash
conan install . --build=missing -s build_type=Debug
cmake --preset conan-debug
cmake --build --preset conan-debug
```

### With Coverage

```bash
conan install . --build=missing -s build_type=Debug
cmake --preset conan-debug -DFULLSCRATCH_ENABLE_COVERAGE=ON
cmake --build --preset conan-debug
ctest --preset conan-debug

# Generate coverage report
lcov --directory . --capture --output-file coverage.info
genhtml coverage.info --output-directory coverage_report
```

### With Sanitizers

```bash
conan install . --build=missing -s build_type=Debug
cmake --preset conan-debug -DFULLSCRATCH_ENABLE_SANITIZERS=ON
cmake --build --preset conan-debug
ctest --preset conan-debug
```

### With Python Bindings

```bash
conan install . --build=missing -s build_type=Release
cmake --preset conan-release -DFULLSCRATCH_BUILD_PYTHON=ON
cmake --build --preset conan-release

# Install Python package
cd python
pip install .
```

## Troubleshooting

### Common Issues

#### 1. Conan Cannot Find Compiler

```bash
# Manually specify compiler
conan profile detect --force
conan profile show default
# Edit ~/.conan2/profiles/default if needed
```

#### 2. CMake Cannot Find Dependencies

Make sure you ran `conan install` before configuring:
```bash
conan install . --build=missing
cmake --preset conan-release
```

#### 3. Tests Fail

```bash
# Run tests with verbose output
ctest --preset conan-release --output-on-failure --verbose
```

#### 4. Linker Errors on Windows

Make sure you're using the same build type for all dependencies:
```bash
conan install . --build=missing -s build_type=Release
cmake --preset conan-release
```

### Platform-Specific Issues

#### macOS: Apple Silicon

```bash
# Ensure correct architecture
conan profile show default
# Should show arch=armv8 or arch=x86_64
```

#### Linux: Missing Libraries

```bash
# Install additional development packages
sudo apt-get install -y libstdc++-11-dev
```

## Clean Build

```bash
# Remove build artifacts
rm -rf build CMakeCache.txt CMakeFiles

# Remove Conan cache (if needed)
rm -rf ~/.conan2/p

# Rebuild
conan install . --build=missing
cmake --preset conan-release
cmake --build --preset conan-release
```

## IDE Integration

### Visual Studio Code

Install extensions:
- C/C++ (Microsoft)
- CMake Tools (Microsoft)
- clangd (optional, for better code intelligence)

Configure `.vscode/settings.json`:
```json
{
    "cmake.configureSettings": {
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
    },
    "C_Cpp.default.compileCommands": "${workspaceFolder}/build/compile_commands.json"
}
```

### CLion

CLion has native CMake support. Simply open the project root directory.

Configure CMake profiles:
- Settings → Build, Execution, Deployment → CMake
- Add profiles for Debug and Release

### Visual Studio

Open the folder in Visual Studio 2022. It will automatically detect CMake configuration.

## Performance Optimization

### Release Build with Maximum Optimization

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -flto"
cmake --build build --parallel
```

### Link-Time Optimization (LTO)

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
cmake --build build --parallel
```
