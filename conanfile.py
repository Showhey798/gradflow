from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps


class GradFlowConan(ConanFile):
    name = "gradflow"
    version = "0.1.0"
    license = "MIT"
    author = "GradFlow Contributors"
    url = "https://github.com/yourusername/fullScratchLibs"
    description = "A comprehensive C++ machine learning library built from scratch"
    topics = ("machine-learning", "deep-learning", "neural-networks", "cpp", "header-only")

    settings = "os", "compiler", "build_type", "arch"

    options = {
        "build_tests": [True, False],
        "build_examples": [True, False],
        "build_benchmarks": [True, False],
        "build_python": [True, False],
        "enable_coverage": [True, False],
        "enable_sanitizers": [True, False],
    }

    default_options = {
        "build_tests": True,
        "build_examples": True,
        "build_benchmarks": False,
        "build_python": False,
        "enable_coverage": False,
        "enable_sanitizers": False,
    }

    exports_sources = "CMakeLists.txt", "src/*", "include/*", "tests/*", "examples/*", "cmake/*"

    def requirements(self):
        # Core dependencies
        self.requires("eigen/3.4.0")

    def build_requirements(self):
        # Test dependencies
        if self.options.build_tests:
            self.test_requires("gtest/1.14.0")

        # Benchmark dependencies
        if self.options.build_benchmarks:
            self.test_requires("benchmark/1.8.3")

        # Python binding dependencies
        if self.options.build_python:
            self.requires("pybind11/2.11.1")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()

        tc = CMakeToolchain(self)
        tc.variables["GRADFLOW_BUILD_TESTS"] = self.options.build_tests
        tc.variables["GRADFLOW_BUILD_EXAMPLES"] = self.options.build_examples
        tc.variables["GRADFLOW_BUILD_BENCHMARKS"] = self.options.build_benchmarks
        tc.variables["GRADFLOW_BUILD_PYTHON"] = self.options.build_python
        tc.variables["GRADFLOW_ENABLE_COVERAGE"] = self.options.enable_coverage
        tc.variables["GRADFLOW_ENABLE_SANITIZERS"] = self.options.enable_sanitizers
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["gradflow"]
        self.cpp_info.includedirs = ["include"]
        self.cpp_info.set_property("cmake_file_name", "GradFlow")
        self.cpp_info.set_property("cmake_target_name", "GradFlow::gradflow")
