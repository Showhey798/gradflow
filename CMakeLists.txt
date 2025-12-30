cmake_minimum_required(VERSION 3.20)

project(
  GradFlow
  VERSION 0.1.0
  DESCRIPTION "A comprehensive C++ machine learning library built from scratch"
  LANGUAGES CXX)

# ========================================
# Project Configuration
# ========================================

# C++ standard requirement
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type default
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ========================================
# Build Options
# ========================================

option(GRADFLOW_BUILD_TESTS "Build tests" ON)
option(GRADFLOW_BUILD_EXAMPLES "Build examples" ON)
option(GRADFLOW_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(GRADFLOW_BUILD_PYTHON "Build Python bindings" OFF)
option(GRADFLOW_ENABLE_COVERAGE "Enable code coverage" OFF)
option(GRADFLOW_ENABLE_SANITIZERS "Enable sanitizers" OFF)
option(GRADFLOW_ENABLE_METAL "Enable Metal GPU backend" OFF)
option(GRADFLOW_ENABLE_CUDA "Enable CUDA GPU backend" OFF)
option(GRADFLOW_ENABLE_MULTI_GPU "Enable Multi-GPU support" OFF)

# ========================================
# Dependencies
# ========================================

# Include custom CMake modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Find or fetch dependencies using Conan or system packages
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, will be fetched via Conan if enabled")
endif()

# ========================================
# Compiler Settings
# ========================================

# Compiler warnings
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Werror
        -Wconversion
        -Wsign-conversion
        -Wcast-qual
        -Wformat=2
        -Wundef
        -Werror=float-equal
    )
endif()

# Optimization flags for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(NOT MSVC)
        add_compile_options(-O3 -march=native)
    endif()
endif()

# Sanitizers (only for Debug builds)
if(GRADFLOW_ENABLE_SANITIZERS AND NOT MSVC)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# Code Coverage (only for Debug builds)
if(GRADFLOW_ENABLE_COVERAGE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(--coverage)
    add_link_options(--coverage)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate -fcoverage-mapping)
  endif()
endif()

# ========================================
# Library Target
# ========================================

add_library(fullscratch INTERFACE)
target_include_directories(
  fullscratch INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                        $<INSTALL_INTERFACE:include>)
target_compile_features(fullscratch INTERFACE cxx_std_17)

# Link dependencies
if(Eigen3_FOUND)
    target_link_libraries(fullscratch INTERFACE Eigen3::Eigen)
endif()

# ========================================
# Subdirectories
# ========================================

# Add source directory (if we have compiled sources later)
# add_subdirectory(src)

# Add tests
if(GRADFLOW_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Add examples
if(GRADFLOW_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Add benchmarks
if(GRADFLOW_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Add Python bindings
if(GRADFLOW_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# ========================================
# Installation
# ========================================

include(GNUInstallDirs)

install(TARGETS fullscratch
    EXPORT GradFlowTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Export CMake targets
install(EXPORT GradFlowTargets
    FILE GradFlowTargets.cmake
    NAMESPACE GradFlow::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GradFlow
)

# Create and install Config file
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/GradFlowConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/GradFlowConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GradFlow
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/GradFlowConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/GradFlowConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/GradFlowConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GradFlow
)

# ========================================
# Summary
# ========================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "GradFlow Configuration Summary")
message(STATUS "========================================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  Build Tests: ${GRADFLOW_BUILD_TESTS}")
message(STATUS "  Build Examples: ${GRADFLOW_BUILD_EXAMPLES}")
message(STATUS "  Build Benchmarks: ${GRADFLOW_BUILD_BENCHMARKS}")
message(STATUS "  Build Python Bindings: ${GRADFLOW_BUILD_PYTHON}")
message(STATUS "  Enable Coverage: ${GRADFLOW_ENABLE_COVERAGE}")
message(STATUS "  Enable Sanitizers: ${GRADFLOW_ENABLE_SANITIZERS}")
message(STATUS "========================================")
message(STATUS "")
