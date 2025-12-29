#!/usr/bin/env bash
# CI Python Lint Script
# This script runs Python code quality checks (ruff, pyright)
# Usage: ./scripts/ci-lint-python.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "==> Running Python code quality checks..."

cd "${PROJECT_ROOT}"

# Check if ruff is available
if ! command -v ruff &> /dev/null; then
    echo "Error: ruff not found"
    echo "Please install ruff: pip install ruff"
    exit 1
fi

# Check if pyright is available
if ! command -v pyright &> /dev/null; then
    echo "Warning: pyright not found, skipping type checking"
    echo "Install pyright for type checking: pip install pyright"
    SKIP_PYRIGHT=1
else
    SKIP_PYRIGHT=0
fi

echo "==> Running ruff format check..."
ruff format --check python

echo "==> Running ruff lint..."
ruff check python

if [ "$SKIP_PYRIGHT" -eq 0 ]; then
    echo "==> Running pyright type check..."
    pyright python
fi

echo "==> Python quality checks passed!"
