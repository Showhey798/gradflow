#include <gradflow/fullscratch.hpp>
#include <iostream>

int main() {
    std::cout << "=== FullScratchML Matrix Operations Example ===" << std::endl;
    std::cout << std::endl;

    // Create matrices
    fullscratch::Matrix<double> a = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    fullscratch::Matrix<double> b = {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};

    std::cout << "Matrix A (2x3):" << std::endl;
    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t j = 0; j < a.cols(); ++j) {
            std::cout << a(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Matrix B (3x2):" << std::endl;
    for (size_t i = 0; i < b.rows(); ++i) {
        for (size_t j = 0; j < b.cols(); ++j) {
            std::cout << b(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Matrix multiplication
    auto c = a * b;

    std::cout << "Matrix C = A * B (2x2):" << std::endl;
    for (size_t i = 0; i < c.rows(); ++i) {
        for (size_t j = 0; j < c.cols(); ++j) {
            std::cout << c(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Matrix transpose
    auto d = a.transpose();

    std::cout << "Matrix D = A^T (3x2):" << std::endl;
    for (size_t i = 0; i < d.rows(); ++i) {
        for (size_t j = 0; j < d.cols(); ++j) {
            std::cout << d(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Scalar multiplication
    auto e = a * 2.0;

    std::cout << "Matrix E = A * 2.0 (2x3):" << std::endl;
    for (size_t i = 0; i < e.rows(); ++i) {
        for (size_t j = 0; j < e.cols(); ++j) {
            std::cout << e(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Identity matrix
    auto identity = fullscratch::Matrix<double>::identity(3);

    std::cout << "Identity Matrix (3x3):" << std::endl;
    for (size_t i = 0; i < identity.rows(); ++i) {
        for (size_t j = 0; j < identity.cols(); ++j) {
            std::cout << identity(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "=== Example completed successfully ===" << std::endl;

    return 0;
}
