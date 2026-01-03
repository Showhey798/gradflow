# Source files for the compiled part of the library
set(GRADFLOW_SOURCES autograd/device.cpp autograd/memory_pool.cpp)

# Metal サポート (macOS のみ)
if(APPLE AND GRADFLOW_ENABLE_METAL)
  message(STATUS "Enabling Metal GPU backend")
  list(APPEND GRADFLOW_SOURCES autograd/metal/device.mm
       autograd/metal/allocator.mm autograd/metal/kernels.mm
       autograd/metal/grad_kernels.mm)
endif()

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

# Metal サポートの設定
if(APPLE AND GRADFLOW_ENABLE_METAL)
  # Metal framework をリンク
  target_link_libraries(
    gradflow_impl PUBLIC "-framework Foundation" "-framework Metal"
                         "-framework MetalPerformanceShaders")

  # Metal を有効化するマクロを定義
  target_compile_definitions(gradflow_impl PUBLIC GRADFLOW_HAS_METAL)

  # Objective-C++ フラグ (ARC を無効化、手動メモリ管理)
  set_source_files_properties(
    autograd/metal/device.mm autograd/metal/allocator.mm
    autograd/metal/kernels.mm autograd/metal/grad_kernels.mm
    PROPERTIES COMPILE_FLAGS "-fno-objc-arc")

  # Metal ソースを事前コンパイルして .metallib を生成（オプション） metal コマンドが利用できる場合のみ実行
  find_program(METAL_COMPILER metal)
  if(METAL_COMPILER)
    set(METAL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/autograd/metal/kernels.metal)
    set(METAL_GRAD_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/autograd/metal/grad_kernels.metal)
    set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/kernels.air)
    set(METAL_GRAD_AIR ${CMAKE_CURRENT_BINARY_DIR}/grad_kernels.air)
    set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/kernels.metallib)
    set(METAL_GRAD_LIB ${CMAKE_CURRENT_BINARY_DIR}/grad_kernels.metallib)

    add_custom_command(
      OUTPUT ${METAL_AIR}
      COMMAND xcrun -sdk macosx metal -c ${METAL_SOURCE} -o ${METAL_AIR}
      DEPENDS ${METAL_SOURCE}
      COMMENT "Compiling Metal shader to AIR: ${METAL_SOURCE}"
      VERBATIM)

    add_custom_command(
      OUTPUT ${METAL_GRAD_AIR}
      COMMAND xcrun -sdk macosx metal -c ${METAL_GRAD_SOURCE} -o
              ${METAL_GRAD_AIR}
      DEPENDS ${METAL_GRAD_SOURCE}
      COMMENT "Compiling Metal gradient shader to AIR: ${METAL_GRAD_SOURCE}"
      VERBATIM)

    add_custom_command(
      OUTPUT ${METAL_LIB}
      COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
      DEPENDS ${METAL_AIR}
      COMMENT "Creating Metal library: ${METAL_LIB}"
      VERBATIM)

    add_custom_command(
      OUTPUT ${METAL_GRAD_LIB}
      COMMAND xcrun -sdk macosx metallib ${METAL_GRAD_AIR} -o ${METAL_GRAD_LIB}
      DEPENDS ${METAL_GRAD_AIR}
      COMMENT "Creating Metal gradient library: ${METAL_GRAD_LIB}"
      VERBATIM)

    add_custom_target(metal_shaders ALL DEPENDS ${METAL_LIB} ${METAL_GRAD_LIB})
    add_dependencies(gradflow_impl metal_shaders)

    # .metallib のパスを C++ に渡す
    target_compile_definitions(
      gradflow_impl PRIVATE GRADFLOW_METAL_LIB_PATH="${METAL_LIB}"
                            GRADFLOW_METAL_GRAD_LIB_PATH="${METAL_GRAD_LIB}")

    # Install .metallib
    install(
      FILES ${METAL_LIB} ${METAL_GRAD_LIB}
      DESTINATION lib
      COMPONENT runtime)

    message(STATUS "Metal shader precompilation enabled (metal compiler found)")
  else()
    message(
      STATUS
        "Metal shader precompilation disabled (metal compiler not found, will use runtime compilation)"
    )
  endif()
endif()
