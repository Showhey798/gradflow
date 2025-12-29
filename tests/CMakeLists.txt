# FullScratchML Test Suite Configuration

# Find or fetch Google Test
find_package(GTest REQUIRED)

# Enable testing
include(GoogleTest)

# Test utilities library (common fixtures, helpers)
add_library(test_utils INTERFACE)
target_include_directories(test_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_utils INTERFACE fullscratch GTest::gtest GTest::gtest_main)

# Test configuration
add_compile_definitions(
    FULLSCRATCH_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

# Helper function to add a test with common settings
function(fullscratch_add_test TEST_NAME)
    cmake_parse_arguments(
        ARG
        ""
        ""
        "SOURCES;LIBS"
        ${ARGN}
    )

    add_executable(${TEST_NAME} ${ARG_SOURCES})

    target_link_libraries(${TEST_NAME}
        PRIVATE
            test_utils
            ${ARG_LIBS}
    )

    # Set random seed for reproducibility
    set_target_properties(${TEST_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    # Add to CTest
    gtest_discover_tests(${TEST_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        PROPERTIES
            TIMEOUT 60
            ENVIRONMENT "GTEST_COLOR=1"
    )

    # Add coverage support if enabled
    if(FULLSCRATCH_ENABLE_COVERAGE)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_options(${TEST_NAME} PRIVATE --coverage)
            target_link_options(${TEST_NAME} PRIVATE --coverage)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(${TEST_NAME} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
            target_link_options(${TEST_NAME} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
        endif()
    endif()
endfunction()

# ========================================
# Core Tests
# ========================================

# Optimizer tests
fullscratch_add_test(optimizer_test
    SOURCES test_optimizer.cpp
)

# TODO: Implement these tests
# # Matrix operations tests
# fullscratch_add_test(matrix_test
#     SOURCES test_matrix.cpp
# )
#
# # Vector operations tests
# fullscratch_add_test(vector_test
#     SOURCES test_vector.cpp
# )
#
# # Activation functions tests
# fullscratch_add_test(activation_test
#     SOURCES test_activation.cpp
# )
#
# # Loss functions tests
# fullscratch_add_test(loss_test
#     SOURCES test_loss.cpp
# )
#
# # Neural network tests
# fullscratch_add_test(neural_network_test
#     SOURCES test_neural_network.cpp
# )
#
# # Data loader tests
# fullscratch_add_test(data_loader_test
#     SOURCES test_data_loader.cpp
# )

# ========================================
# Integration Tests
# ========================================

# TODO: Implement integration tests
# fullscratch_add_test(integration_test
#     SOURCES test_integration.cpp
# )

# ========================================
# Custom Test Target
# ========================================

# Add custom target to run all tests with verbose output
add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
    DEPENDS
        optimizer_test
        # matrix_test
        # vector_test
        # activation_test
        # loss_test
        # neural_network_test
        # data_loader_test
        # integration_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all tests with verbose output"
)
