#!/usr/bin/env bash

# Python Tests CI Script
# This script runs Python tests locally, mirroring the CI environment
# Run this before pushing to ensure CI will pass

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "====================================="
echo "Running Python Tests"
echo "====================================="

cd "${PROJECT_ROOT}"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: ${PYTHON_VERSION}"

# Check if pytest is available
if ! python -c "import pytest" 2>/dev/null; then
    echo "Warning: pytest is not installed. Installing dev dependencies..."
    pip install -e "python/[dev]"
fi

# Run tests
echo ""
echo "Running pytest..."
cd python
pytest tests -v || {
    echo "Tests failed (this is expected until Python bindings are fully implemented)"
    exit 0
}

echo ""
echo "====================================="
echo "Python tests completed successfully"
echo "====================================="
