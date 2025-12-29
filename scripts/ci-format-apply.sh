#!/usr/bin/env bash
# CI Format Apply Script
# This script applies clang-format to all C++ files
# Usage: ./scripts/ci-format-apply.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "==> Applying clang-format to C++ files..."

cd "${PROJECT_ROOT}"

# Find all C++ files
CPP_FILES=$(find include src tests examples -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) 2>/dev/null || true)

if [ -z "$CPP_FILES" ]; then
    echo "No C++ source files found, skipping clang-format"
    exit 0
fi

# Check if clang-format is available
if ! command -v clang-format &> /dev/null; then
    echo "Error: clang-format not found"
    echo "Please install clang-format (version 15 or later recommended)"
    exit 1
fi

echo "Applying clang-format to $(echo "$CPP_FILES" | wc -l) files..."

# Apply formatting
echo "$CPP_FILES" | xargs clang-format -i -style=file

echo "==> Formatting applied successfully!"
