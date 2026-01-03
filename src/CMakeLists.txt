# Source files for the compiled part of the library
set(GRADFLOW_SOURCES autograd/device.cpp)

# Metal サポート (macOS のみ)
if(APPLE AND GRADFLOW_ENABLE_METAL)
  message(STATUS "Enabling Metal GPU backend")
  list(APPEND GRADFLOW_SOURCES autograd/metal/device.mm
       autograd/metal/allocator.mm autograd/metal/kernels.mm)
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
    autograd/metal/kernels.mm PROPERTIES COMPILE_FLAGS "-fno-objc-arc")

  # Metal shader source のインストール (runtime でコンパイル) Note: xcrun metal
  # が利用できない環境では、runtime で .metal ファイルから library を作成します install(FILES
  # ${CMAKE_CURRENT_SOURCE_DIR}/autograd/metal/kernels.metal DESTINATION lib
  # COMPONENT runtime)
endif()
