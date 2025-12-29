#!/usr/bin/env bash
# CI C++ Lint Script
# This script runs clang-tidy on C++ source files
# Usage: ./scripts/ci-lint-cpp.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "==> Running clang-tidy on C++ files..."

cd "${PROJECT_ROOT}"

# Check if clang-tidy is available
if ! command -v clang-tidy &> /dev/null; then
    echo "Error: clang-tidy not found"
    echo "Please install clang-tidy (version 15 or later recommended)"
    exit 1
fi

# Find compile_commands.json
COMPILE_COMMANDS=$(find build -name compile_commands.json 2>/dev/null | head -n 1 || true)

if [ -z "$COMPILE_COMMANDS" ]; then
    echo "Error: compile_commands.json not found in build directory"
    echo "Please configure and build the project first:"
    echo "  conan install . --output-folder=build --build=missing -s build_type=Debug"
    echo "  cmake --preset conan-debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    exit 1
fi

BUILD_DIR="$(dirname "$COMPILE_COMMANDS")"
echo "Using compilation database: $COMPILE_COMMANDS"

# Find C++ source files
CPP_FILES=$(find tests include -type f \( -name '*.cpp' -o -name '*.hpp' \) 2>/dev/null || true)

if [ -z "$CPP_FILES" ]; then
    echo "No C++ source files found, skipping clang-tidy check"
    exit 0
fi

echo "Checking $(echo "$CPP_FILES" | wc -l) files..."

# Run clang-tidy
echo "$CPP_FILES" | xargs clang-tidy \
    -p "$BUILD_DIR" \
    --config-file=.clang-tidy

echo "==> clang-tidy check passed!"
