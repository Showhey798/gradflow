#!/usr/bin/env bash
# CMake Format Check Script
# This script checks CMake files formatting with cmake-format
# Usage: ./scripts/ci-format-cmake.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "==> Checking CMake files formatting..."

cd "${PROJECT_ROOT}"

# Check if cmake-format is available
if ! command -v cmake-format &> /dev/null; then
    echo "Error: cmake-format not found"
    echo "Please install cmake-format: pip install cmakelang"
    exit 1
fi

# Find all CMake files
CMAKE_FILES=$(find . -name 'CMakeLists.txt' -o -name '*.cmake' 2>/dev/null | grep -v build || true)

if [ -z "$CMAKE_FILES" ]; then
    echo "No CMake files found"
    exit 0
fi

echo "Found $(echo "$CMAKE_FILES" | wc -l) CMake files to check"

# Create a temporary directory for formatted files
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Format files to temp directory and compare
HAS_DIFF=false
while IFS= read -r file; do
    cmake-format "$file" > "$TEMP_DIR/$(basename "$file")"
    if ! diff -q "$file" "$TEMP_DIR/$(basename "$file")" > /dev/null 2>&1; then
        echo "❌ $file needs formatting"
        HAS_DIFF=true
    fi
done <<< "$CMAKE_FILES"

if [ "$HAS_DIFF" = true ]; then
    echo ""
    echo "CMake files need formatting. Run: cmake-format -i <file>"
    echo "Or run: find . -name 'CMakeLists.txt' -o -name '*.cmake' | xargs cmake-format -i"
    exit 1
fi

echo "==> CMake format check passed!"
