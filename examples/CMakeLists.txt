# FullScratchML Examples

# Helper function to add an example
function(fullscratch_add_example EXAMPLE_NAME)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})

  add_executable(${EXAMPLE_NAME} ${ARG_SOURCES})

  target_link_libraries(${EXAMPLE_NAME} PRIVATE fullscratch ${ARG_LIBS})

  set_target_properties(
    ${EXAMPLE_NAME}
    PROPERTIES CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF)

  # Install examples (optional)
  install(TARGETS ${EXAMPLE_NAME}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples)
endfunction()

# ========================================
# Examples
# ========================================

# TODO: Implement examples # Basic matrix operations
# fullscratch_add_example(matrix_operations SOURCES matrix_operations.cpp )
#
# # Simple neural network fullscratch_add_example(simple_neural_network SOURCES
# simple_neural_network.cpp )

# Custom target to build all examples
add_custom_target(
  examples
  # DEPENDS
  # matrix_operations
  # simple_neural_network
  COMMENT "Building all examples")
