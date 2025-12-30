# Source files for the compiled part of the library
set(GRADFLOW_SOURCES autograd/device.cpp)

# Create a static library for compiled sources
add_library(gradflow_impl STATIC ${GRADFLOW_SOURCES})

# Set include directories
target_include_directories(
  gradflow_impl PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                       $<INSTALL_INTERFACE:include>)

# Set C++ standard
target_compile_features(gradflow_impl PUBLIC cxx_std_17)

# Link dependencies
if(Eigen3_FOUND)
  target_link_libraries(gradflow_impl PUBLIC Eigen3::Eigen)
endif()
