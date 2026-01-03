# FullScratchML Test Suite Configuration

# Find or fetch Google Test
find_package(GTest REQUIRED)

# Enable testing
include(GoogleTest)

# Test utilities library (common fixtures, helpers)
add_library(test_utils INTERFACE)
target_include_directories(test_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_utils INTERFACE fullscratch gradflow_impl
                                           GTest::gtest GTest::gtest_main)

# Test configuration
add_compile_definitions(
  FULLSCRATCH_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

# Helper function to add a test with common settings
function(fullscratch_add_test TEST_NAME)
  cmake_parse_arguments(ARG "" "TIMEOUT" "SOURCES;LIBS" ${ARGN})

  add_executable(${TEST_NAME} ${ARG_SOURCES})

  target_link_libraries(${TEST_NAME} PRIVATE test_utils ${ARG_LIBS})

  # Set random seed for reproducibility
  set_target_properties(
    ${TEST_NAME}
    PROPERTIES CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF)

  # Set timeout value (default 60 seconds)
  if(NOT ARG_TIMEOUT)
    set(ARG_TIMEOUT 60)
  endif()

  # Add to CTest
  gtest_discover_tests(
    ${TEST_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES
    TIMEOUT ${ARG_TIMEOUT} ENVIRONMENT "GTEST_COLOR=1")

  # Add coverage support if enabled
  if(FULLSCRATCH_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      target_compile_options(${TEST_NAME} PRIVATE --coverage)
      target_link_options(${TEST_NAME} PRIVATE --coverage)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      target_compile_options(${TEST_NAME} PRIVATE -fprofile-instr-generate
                                                  -fcoverage-mapping)
      target_link_options(${TEST_NAME} PRIVATE -fprofile-instr-generate
                          -fcoverage-mapping)
    endif()
  endif()
endfunction()

# ========================================
# Core Tests
# ========================================

# Shape tests
fullscratch_add_test(shape_test SOURCES test_shape.cpp)

# Storage tests
fullscratch_add_test(storage_test SOURCES test_storage.cpp)

# Tensor tests
fullscratch_add_test(tensor_test SOURCES test_tensor.cpp)

# Optimizer tests
fullscratch_add_test(optimizer_test SOURCES test_optimizer.cpp)

# Operations tests
fullscratch_add_test(ops_test SOURCES test_ops.cpp)

# Device tests
fullscratch_add_test(device_test SOURCES test_device.cpp)

# Phase 3: Metal GPU tests (macOS only)
if(APPLE AND GRADFLOW_ENABLE_METAL)
  fullscratch_add_test(metal_device_test SOURCES test_metal_device.cpp)
endif()

# Operation tests
fullscratch_add_test(operation_test SOURCES test_operation.cpp)

# Variable tests
fullscratch_add_test(variable_test SOURCES test_variable.cpp)

# Operations gradient tests
fullscratch_add_test(ops_grad_test SOURCES test_ops_grad.cpp)

# Activation operations tests
fullscratch_add_test(activation_ops_test SOURCES test_activation_ops.cpp)

# Loss operations tests
fullscratch_add_test(loss_grad_test SOURCES test_loss_grad.cpp)

# Phase 1 integration tests Set longer timeout (300s) to accommodate sanitizer
# overhead Sanitizers can add 10-50x overhead, especially for large scale
# operations
fullscratch_add_test(phase1_integration_test SOURCES
                     test_phase1_integration.cpp TIMEOUT 300)

# Phase 2 integration tests Set longer timeout (300s) to accommodate sanitizer
# overhead and XOR training iterations
fullscratch_add_test(phase2_integration_test SOURCES
                     test_phase2_integration.cpp TIMEOUT 300)

# TODO: Implement these tests # Matrix operations tests
# fullscratch_add_test(matrix_test SOURCES test_matrix.cpp )
#
# # Vector operations tests fullscratch_add_test(vector_test SOURCES
# test_vector.cpp )
#
# # Activation functions tests fullscratch_add_test(activation_test SOURCES
# test_activation.cpp )
#
# # Loss functions tests fullscratch_add_test(loss_test SOURCES test_loss.cpp )
#
# # Neural network tests fullscratch_add_test(neural_network_test SOURCES
# test_neural_network.cpp )
#
# # Data loader tests fullscratch_add_test(data_loader_test SOURCES
# test_data_loader.cpp )

# ========================================
# Integration Tests
# ========================================

# Phase 1 integration tests are defined above with core tests TODO: Implement
# additional integration tests as needed fullscratch_add_test(integration_test
# SOURCES test_integration.cpp )

# ========================================
# Custom Test Target
# ========================================

# Add custom target to run all tests with verbose output
set(ALL_TESTS
    shape_test
    storage_test
    tensor_test
    optimizer_test
    ops_test
    device_test
    operation_test
    variable_test
    ops_grad_test
    activation_ops_test
    loss_grad_test
    phase1_integration_test
    phase2_integration_test)

# Add Metal tests if enabled
if(APPLE AND GRADFLOW_ENABLE_METAL)
  list(APPEND ALL_TESTS metal_device_test)
endif()

add_custom_target(
  check
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
  DEPENDS ${ALL_TESTS}
          # matrix_test vector_test activation_test loss_test
          # neural_network_test data_loader_test
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running all tests with verbose output")
