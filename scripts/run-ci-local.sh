#!/usr/bin/env bash
# Local CI Execution Script
# This script runs all CI checks locally before pushing to remote
# Usage: ./scripts/run-ci-local.sh [--quick] [--skip-tests]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse arguments
QUICK_MODE=false
SKIP_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick] [--skip-tests]"
            exit 1
            ;;
    esac
done

cd "${PROJECT_ROOT}"

echo "========================================="
echo "Running Local CI Checks"
echo "========================================="
echo ""

# Track failures
FAILED_CHECKS=()

run_check() {
    local check_name="$1"
    local check_command="$2"

    echo ">>> Running: $check_name"
    if eval "$check_command"; then
        echo "✅ $check_name passed"
    else
        echo "❌ $check_name failed"
        FAILED_CHECKS+=("$check_name")
    fi
    echo ""
}

# 1. CMake Format Check
run_check "CMake Format Check" "bash scripts/ci-format-cmake.sh"

# 2. C++ Format Check
run_check "C++ Format Check" "bash scripts/ci-format-check.sh"

# 3. Python Format Check (if Python files exist)
if [ -d "python" ]; then
    run_check "Python Format Check" "bash scripts/ci-lint-python.sh"
fi

# 4. C++ Lint (clang-tidy)
if [ "$QUICK_MODE" = false ]; then
    run_check "C++ Lint (clang-tidy)" "bash scripts/ci-lint-cpp.sh"
fi

# 5. Build
run_check "Build" "bash scripts/build.sh"

# 6. Tests
if [ "$SKIP_TESTS" = false ]; then
    run_check "C++ Tests" "bash scripts/ci-test-cpp.sh"

    if [ -d "python" ]; then
        run_check "Python Tests" "bash scripts/ci-test-python.sh"
    fi
fi

# Summary
echo "========================================="
echo "Local CI Check Summary"
echo "========================================="

if [ ${#FAILED_CHECKS[@]} -eq 0 ]; then
    echo "✅ All checks passed!"
    echo ""
    echo "You can safely push your changes."
    exit 0
else
    echo "❌ ${#FAILED_CHECKS[@]} check(s) failed:"
    for check in "${FAILED_CHECKS[@]}"; do
        echo "  - $check"
    done
    echo ""
    echo "Please fix the issues before pushing."
    exit 1
fi
