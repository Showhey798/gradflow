# FullScratchML Python Bindings

# Find Python and pybind11
find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED)
find_package(pybind11 REQUIRED)

# Python module
pybind11_add_module(
  pyfullscratch
  bindings.cpp
  matrix_bindings.cpp
  vector_bindings.cpp
  activation_bindings.cpp
  loss_bindings.cpp
  neural_network_bindings.cpp
  optimizer_bindings.cpp)

target_link_libraries(pyfullscratch PRIVATE fullscratch)

# Set module properties
set_target_properties(
  pyfullscratch
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF
             OUTPUT_NAME "fullscratch")

# Installation
install(TARGETS pyfullscratch
        LIBRARY DESTINATION ${Python_SITEARCH}/fullscratch)

# Install Python package files
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/fullscratch/__init__.py
        DESTINATION ${Python_SITEARCH}/fullscratch)
