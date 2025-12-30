#!/usr/bin/env bash
# CI Verification Script
# This script runs all CI checks locally
# Usage: ./scripts/ci-verify.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=========================================="
echo "CI Verification (Local)"
echo "=========================================="
echo ""

# Track failures
FAILED=0

# Function to run a check
run_check() {
    local name=$1
    local script=$2
    echo ""
    echo "=========================================="
    echo "Running: $name"
    echo "=========================================="
    if bash "$script"; then
        echo "✓ $name passed"
    else
        echo "✗ $name failed"
        FAILED=$((FAILED + 1))
    fi
}

# Run all checks
run_check "C++ Format Check" "${SCRIPT_DIR}/ci-format-check.sh"
run_check "Python Lint" "${SCRIPT_DIR}/ci-lint-python.sh"
run_check "C++ Tests" "${SCRIPT_DIR}/ci-test-cpp.sh"
run_check "Python Tests" "${SCRIPT_DIR}/ci-test-python.sh"

# Report results
echo ""
echo "=========================================="
echo "Verification Complete"
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo "✓ All checks passed!"
    exit 0
else
    echo "✗ $FAILED check(s) failed"
    exit 1
fi
