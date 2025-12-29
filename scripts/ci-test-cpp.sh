#!/usr/bin/env bash
# CI C++ Test Script
# This script builds and runs C++ tests
# Usage: ./scripts/ci-test-cpp.sh [preset]
#   preset: CMake preset name (default: conan-release)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PRESET="${1:-conan-release}"

echo "==> Building and testing C++ with preset: $PRESET"

cd "${PROJECT_ROOT}"

# Ensure dependencies are installed
if [ ! -d "build" ] || [ ! -f "build/conan_toolchain.cmake" ]; then
    echo "==> Installing dependencies with Conan..."
    BUILD_TYPE="Release"
    if [[ "$PRESET" == *"debug"* ]]; then
        BUILD_TYPE="Debug"
    fi
    conan install . --output-folder=build --build=missing -s build_type="$BUILD_TYPE"
fi

# Configure
echo "==> Configuring CMake..."
cmake --preset "$PRESET"

# Build
echo "==> Building project..."
cmake --build --preset "$PRESET" --parallel

# Test
echo "==> Running tests..."
ctest --preset "$PRESET" --output-on-failure

echo "==> C++ tests passed!"
