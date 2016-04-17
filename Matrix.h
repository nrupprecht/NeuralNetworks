// 
// A wrapper class for using cblas in a simpler way
//

#ifndef MATRIX_H
#define MATRIX_H

#include "Utility.h"

class Matrix {
 public:
  Matrix();
  Matrix(int rows, int columns);
  Matrix(int rows, int columns, bool rand);
  Matrix(const Matrix& M);
  ~Matrix();
  
  Matrix operator=(const Matrix& M);

  /// Arithmetic functions
  friend void multiply(const Matrix& A, const Matrix& B, Matrix& C);
  friend void multiply(const double m, const Matrix& A, Matrix& B);
  friend void timesEq(Matrix& A, const double m);
  friend void add(const Matrix& A, const Matrix& B, Matrix& C);
  friend void plusEq(Matrix& A, const Matrix& B);
  friend void NTplusEqUnsafe(Matrix& A, const Matrix& B, double mult=1.);
  friend void subtract(const Matrix& A, const Matrix& B, Matrix& C);
  friend void minusEq(Matrix& A, const Matrix& B);
  friend void NTminusEqUnsafe(Matrix& A, const Matrix& B, double mult=1.);
  friend void hadamard(const Matrix& A, const Matrix& B, Matrix& C);
  friend void apply(const Matrix& A, function F, Matrix& C); // Apply F componentwise to A

  /// Error classes
  class MatrixOutOfBounds {};
  class MatrixMismatch {};
  class Unallocated {};
  class BadReshape {};

  /// Accessors
  double& at(int row, int col);
  double at(int row, int col) const;
  double& operator() (int row, int col);
  double operator() (int row, int col) const;
  double& access_NT(int row, int col);
  double& access_T(int row, int col);
  double operator[](int index);
  int getRows() const { return trans ? cols : rows; }
  int getCols() const { return trans ? rows : cols; }
  int getARows() const { return rows; }
  int getACols() const { return cols; }
  int size() const { return rows*cols; }
  double norm() const;
  double max() const;
  double min() const;

  /// Setting Matrices
  void resize(int rows, int cols); // Resize the matrix
  void reshape(int rows, int cols);  // Interpret the matrix as having a different shape
  void random(double max=1);
  void T() { trans = !trans; }
  void zero();
  
  /// Quick handling of matrices
  void qrel();          // Release array memory
  void qref(Matrix& M); // Reference this matrices' array

  /// Printing and reading
  friend std::ostream& operator<<(std::ostream& out, const Matrix& M);
  friend std::istream& operator>>(std::istream& in, Matrix& M);

 private:
  /// Private helper functions
  static inline bool checkDims(const Matrix& A, const Matrix& B);

  /// Private data
  double *array;
  int rows, cols;
  bool trans;
};

#endif
