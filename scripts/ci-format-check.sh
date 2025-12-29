#!/usr/bin/env bash
# CI Format Check Script
# This script checks C++ code formatting with clang-format
# Usage: ./scripts/ci-format-check.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "==> Checking C++ code formatting with clang-format..."

cd "${PROJECT_ROOT}"

# Find all C++ files
CPP_FILES=$(find include src tests examples -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) 2>/dev/null || true)

if [ -z "$CPP_FILES" ]; then
    echo "No C++ source files found, skipping clang-format check"
    exit 0
fi

# Check if clang-format is available
if ! command -v clang-format &> /dev/null; then
    echo "Error: clang-format not found"
    echo "Please install clang-format (version 15 or later recommended)"
    exit 1
fi

echo "Found $(echo "$CPP_FILES" | wc -l) C++ files to check"

# Run clang-format in dry-run mode
echo "$CPP_FILES" | xargs clang-format --dry-run --Werror

echo "==> Format check passed!"
