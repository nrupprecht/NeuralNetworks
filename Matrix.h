// 
// A wrapper class for using cblas in a simpler way
//

#ifndef MATRIX_H
#define MATRIX_H

#include"/afs/crc.nd.edu/x86_64_linux/intel/15.0/mkl/include/mkl.h"

#include <stdlib.h> // For drand48
#include <iostream> // For debugging
using std::cout;
using std::endl;

typedef double (*function) (double);

class Matrix {
 public:
  Matrix();
  Matrix(int rows, int columns);
  Matrix(const Matrix& M);
  ~Matrix();
  
  Matrix operator=(const Matrix& M);

  // Arithmetic functions
  friend void multiply(const Matrix& A, const Matrix& B, Matrix& C);
  friend void multiply(const double m, const Matrix& A, Matrix& B);
  friend void add(const Matrix& A, const Matrix& B, Matrix& C);
  friend void subtract(const Matrix& A, const Matrix& B, Matrix& C);
  friend void hadamard(const Matrix& A, const Matrix& B, Matrix& C);
  friend void apply(const Matrix& A, function F, Matrix& C); // Apply F componentwise to A

  class MatrixOutOfBounds {};
  class MatrixMismatch {};
  class Unallocated {};

  // Accessors
  double& at(int row, int col);
  double at(int row, int col) const;
  double operator[](int index);
  int getRows() const { return trans ? cols : rows; }
  int getCols() const { return trans ? rows : cols; }
  int getARows() const { return rows; }
  int getACols() const { return cols; }

  // Setting Matrices
  void resize(int rows, int cols); // Resize the matrix
  void random(double max=1);
  void T() { trans = !trans; }
  
  // Quick handling of matrices
  void qrel();          // Release array memory
  void qref(Matrix& M); // Reference this matrices' array

  // Printing
  friend std::ostream& operator<<(std::ostream& out, const Matrix& M);

 private:
  // Private helper functions
  static inline bool checkDims(const Matrix& A, const Matrix& B);

  // Private data
  double *array;
  int rows, cols;
  bool trans;
};

#endif