#!/usr/bin/env bash
set -euo pipefail

# FullScratchML Build Script
# This script automates the build process

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TESTS="${BUILD_TESTS:-ON}"
BUILD_EXAMPLES="${BUILD_EXAMPLES:-ON}"
ENABLE_COVERAGE="${ENABLE_COVERAGE:-OFF}"
ENABLE_SANITIZERS="${ENABLE_SANITIZERS:-OFF}"
USE_CONAN="${USE_CONAN:-ON}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --coverage)
            ENABLE_COVERAGE="ON"
            BUILD_TYPE="Debug"
            shift
            ;;
        --sanitizers)
            ENABLE_SANITIZERS="ON"
            BUILD_TYPE="Debug"
            shift
            ;;
        --no-tests)
            BUILD_TESTS="OFF"
            shift
            ;;
        --no-examples)
            BUILD_EXAMPLES="OFF"
            shift
            ;;
        --no-conan)
            USE_CONAN="OFF"
            shift
            ;;
        --clean)
            print_info "Cleaning build directory..."
            rm -rf "$PROJECT_ROOT/$BUILD_DIR"
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug              Build in Debug mode"
            echo "  --release            Build in Release mode (default)"
            echo "  --coverage           Enable code coverage (implies --debug)"
            echo "  --sanitizers         Enable sanitizers (implies --debug)"
            echo "  --no-tests           Don't build tests"
            echo "  --no-examples        Don't build examples"
            echo "  --no-conan           Don't use Conan for dependencies"
            echo "  --clean              Clean build directory before building"
            echo "  -j, --jobs N         Number of parallel jobs (default: auto-detect)"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

print_section "GradFlow Build Configuration"
echo "Build Type:        $BUILD_TYPE"
echo "Build Tests:       $BUILD_TESTS"
echo "Build Examples:    $BUILD_EXAMPLES"
echo "Enable Coverage:   $ENABLE_COVERAGE"
echo "Enable Sanitizers: $ENABLE_SANITIZERS"
echo "Use Conan:         $USE_CONAN"
echo "Parallel Jobs:     $JOBS"
echo ""

# Step 1: Install dependencies with Conan
if [ "$USE_CONAN" = "ON" ]; then
    print_section "Step 1: Installing Dependencies with Conan"

    if ! command -v conan &> /dev/null; then
        print_error "Conan not found. Please install Conan:"
        echo "  pip install conan==2.0.17"
        exit 1
    fi

    print_info "Conan version: $(conan --version)"

    # Detect profile if it doesn't exist
    if [ ! -f "$HOME/.conan2/profiles/default" ]; then
        print_info "Detecting Conan profile..."
        conan profile detect --force
    fi

    print_info "Installing dependencies..."
    conan install . --build=missing -s build_type="$BUILD_TYPE" || {
        print_error "Conan install failed"
        exit 1
    }
else
    print_warn "Skipping Conan dependency installation"
fi

# Step 2: Configure CMake
print_section "Step 2: Configuring CMake"

CMAKE_ARGS=(
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    "-DFULLSCRATCH_BUILD_TESTS=$BUILD_TESTS"
    "-DFULLSCRATCH_BUILD_EXAMPLES=$BUILD_EXAMPLES"
    "-DFULLSCRATCH_ENABLE_COVERAGE=$ENABLE_COVERAGE"
    "-DFULLSCRATCH_ENABLE_SANITIZERS=$ENABLE_SANITIZERS"
)

if [ "$USE_CONAN" = "ON" ]; then
    # Use Conan preset
    PRESET_NAME="conan-$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')"
    print_info "Using CMake preset: $PRESET_NAME"

    cmake --preset "$PRESET_NAME" "${CMAKE_ARGS[@]}" || {
        print_error "CMake configuration failed"
        exit 1
    }
else
    # Manual configuration
    print_info "Configuring manually..."
    mkdir -p "$BUILD_DIR"
    cmake -B "$BUILD_DIR" "${CMAKE_ARGS[@]}" || {
        print_error "CMake configuration failed"
        exit 1
    }
fi

# Step 3: Build
print_section "Step 3: Building"

if [ "$USE_CONAN" = "ON" ]; then
    cmake --build --preset "$PRESET_NAME" --parallel "$JOBS" || {
        print_error "Build failed"
        exit 1
    }
else
    cmake --build "$BUILD_DIR" --parallel "$JOBS" || {
        print_error "Build failed"
        exit 1
    }
fi

# Step 4: Run tests
if [ "$BUILD_TESTS" = "ON" ]; then
    print_section "Step 4: Running Tests"

    if [ "$USE_CONAN" = "ON" ]; then
        ctest --preset "$PRESET_NAME" --output-on-failure || {
            print_error "Tests failed"
            exit 1
        }
    else
        ctest --test-dir "$BUILD_DIR" --output-on-failure || {
            print_error "Tests failed"
            exit 1
        }
    fi
fi

# Step 5: Coverage report
if [ "$ENABLE_COVERAGE" = "ON" ]; then
    print_section "Step 5: Generating Coverage Report"

    if command -v lcov &> /dev/null; then
        print_info "Generating coverage report..."
        lcov --directory . --capture --output-file coverage.info
        lcov --remove coverage.info '/usr/*' '*/tests/*' '*/examples/*' --output-file coverage.info
        lcov --list coverage.info

        if command -v genhtml &> /dev/null; then
            genhtml coverage.info --output-directory coverage_report
            print_info "Coverage report generated in coverage_report/"
        fi
    else
        print_warn "lcov not found, skipping coverage report generation"
    fi
fi

print_section "Build Successful!"

if [ "$BUILD_EXAMPLES" = "ON" ]; then
    print_info "Run example:"
    if [ "$USE_CONAN" = "ON" ]; then
        echo "  ./build/$BUILD_TYPE/examples/matrix_operations"
    else
        echo "  ./$BUILD_DIR/examples/matrix_operations"
    fi
fi

if [ "$BUILD_TESTS" = "ON" ]; then
    print_info "Run tests:"
    if [ "$USE_CONAN" = "ON" ]; then
        echo "  ctest --preset $PRESET_NAME"
    else
        echo "  ctest --test-dir $BUILD_DIR"
    fi
fi
