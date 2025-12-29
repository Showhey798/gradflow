#ifndef GRADFLOW_MATRIX_HPP
#define GRADFLOW_MATRIX_HPP

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace gradflow {

/**
 * @brief A templated matrix class for linear algebra operations
 *
 * This class provides basic matrix operations including addition,
 * multiplication, and transposition. It serves as the foundation
 * for neural network computations.
 *
 * @tparam T The data type (e.g., float, double)
 */
template <typename T>
class Matrix {
  public:
    /**
     * @brief Default constructor
     */
    Matrix() : rows_(0), cols_(0) {}

    /**
     * @brief Constructor with dimensions
     * @param rows Number of rows
     * @param cols Number of columns
     */
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data_(rows * cols) {}

    /**
     * @brief Constructor from initializer list
     * @param init 2D initializer list
     */
    Matrix(std::initializer_list<std::initializer_list<T>> init) {
        rows_ = init.size();
        cols_ = init.begin()->size();
        data_.reserve(rows_ * cols_);

        for (const auto& row : init) {
            for (const auto& val : row) {
                data_.push_back(val);
            }
        }
    }

    /**
     * @brief Get number of rows
     * @return Number of rows
     */
    size_t rows() const { return rows_; }

    /**
     * @brief Get number of columns
     * @return Number of columns
     */
    size_t cols() const { return cols_; }

    /**
     * @brief Element access operator
     * @param row Row index
     * @param col Column index
     * @return Reference to element
     */
    T& operator()(size_t row, size_t col) { return data_[row * cols_ + col]; }

    /**
     * @brief Element access operator (const version)
     * @param row Row index
     * @param col Column index
     * @return Const reference to element
     */
    const T& operator()(size_t row, size_t col) const { return data_[row * cols_ + col]; }

    /**
     * @brief Matrix addition
     * @param other Matrix to add
     * @return Result matrix
     */
    Matrix<T> operator+(const Matrix<T>& other) const {
        Matrix<T> result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    /**
     * @brief Matrix multiplication
     * @param other Matrix to multiply with
     * @return Result matrix
     */
    Matrix<T> operator*(const Matrix<T>& other) const {
        Matrix<T> result(rows_, other.cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < other.cols_; ++j) {
                T sum = 0;
                for (size_t k = 0; k < cols_; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    /**
     * @brief Scalar multiplication
     * @param scalar Scalar value
     * @return Result matrix
     */
    Matrix<T> operator*(T scalar) const {
        Matrix<T> result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    /**
     * @brief Matrix transpose
     * @return Transposed matrix
     */
    Matrix<T> transpose() const {
        Matrix<T> result(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    /**
     * @brief Create zero matrix
     * @param rows Number of rows
     * @param cols Number of columns
     * @return Zero matrix
     */
    static Matrix<T> zeros(size_t rows, size_t cols) {
        Matrix<T> result(rows, cols);
        for (auto& val : result.data_) {
            val = T(0);
        }
        return result;
    }

    /**
     * @brief Create ones matrix
     * @param rows Number of rows
     * @param cols Number of columns
     * @return Ones matrix
     */
    static Matrix<T> ones(size_t rows, size_t cols) {
        Matrix<T> result(rows, cols);
        for (auto& val : result.data_) {
            val = T(1);
        }
        return result;
    }

    /**
     * @brief Create identity matrix
     * @param size Size of square matrix
     * @return Identity matrix
     */
    static Matrix<T> identity(size_t size) {
        Matrix<T> result(size, size);
        for (size_t i = 0; i < size; ++i) {
            result(i, i) = T(1);
        }
        return result;
    }

  private:
    size_t rows_;
    size_t cols_;
    std::vector<T> data_;
};

}  // namespace gradflow

#endif  // GRADFLOW_MATRIX_HPP
