#include "Matrix.h"

Matrix::Matrix() : rows(0), cols(0), array(0), trans(false) {};

Matrix::Matrix(int rows, int columns) 
  : rows(rows), cols(columns), trans(false) {
  array = new double[rows*cols];
  for (int i=0; i<rows*cols; i++) array[i]=0;
}

Matrix::Matrix(const Matrix& M) : rows(M.rows), cols(M.cols), trans(M.trans) {
  array = new double[rows*cols];
  for (int i=0; i<rows*cols; i++)
    array[i] = M.array[i];
}

Matrix::~Matrix() {
  if (array) delete [] array;
  array = 0;
}

Matrix Matrix::operator=(const Matrix& M) {
  if (array) delete [] array;
  rows = M.rows; cols = M.cols; 
  trans = M.trans;
  array = new double[rows*cols];
  for (int i=0;i<rows*cols; i++)
    array[i] = M.array[i];
}

void multiply(const Matrix& A, const Matrix& B, Matrix& C) {
  //C.setT(false);
  if (A.getCols()!=B.getRows() || A.getRows()!=C.getRows() || B.getCols()!=C.getCols())
    throw Matrix::MatrixMismatch();

  auto AT = A.trans ? CblasTrans : CblasNoTrans;
  auto BT = B.trans ? CblasTrans : CblasNoTrans;
  double ALPHA = 1, BETA = 0;
  cblas_dgemm(CblasRowMajor, AT, BT, A.getRows(), B.getCols(), A.getCols(), ALPHA, A.array, A.getACols(), B.array, B.getACols(), BETA, C.array, C.getCols());
}

void multiply(const double m, const Matrix& A, Matrix& B) {
  if (A.getRows()!=B.getRows() || A.getCols()!=B.getCols())
    throw Matrix::MatrixMismatch();
  for (int i=0; i<A.rows*A.cols; i++) 
    B.array[i] = m*A.array[i];
}

void add(const Matrix& A, const Matrix& B, Matrix& C) {
  if (!Matrix::checkDims(A, B) || !Matrix::checkDims(A, C))
    throw Matrix::MatrixMismatch();
  for (int y=0; y<A.getRows(); y++)
    for (int x=0; x<A.getCols(); x++)
      C.at(y,x) = A.at(y,x)+B.at(y,x);
}

void subtract(const Matrix& A, const Matrix& B, Matrix& C) {
  if (!Matrix::checkDims(A, B) || !Matrix::checkDims(A,C))
    throw Matrix::MatrixMismatch();
  for (int y=0;y<A.getRows(); y++)
    for (int x=0; x<A.getCols(); x++)
      C.at(y,x) = A.at(y,x)-B.at(y,x);
}

void hadamard(const Matrix& A, const Matrix& B, Matrix& C) {
  if (!Matrix::checkDims(A,B)) throw Matrix::MatrixMismatch();
  if (A.array==0 || B.array==0 || C.array==0) throw Matrix::Unallocated(); // We could also have an impropper qref
  // Will only work if none are transposed
  for (int i=0; i<A.rows*A.cols; i++) C.array[i] = A.array[i]*B.array[i];
}

void apply(const Matrix& A, function F, Matrix& C) {
  if (A.getCols()!=C.getCols() || A.getRows()!=C.getRows())
    throw Matrix::MatrixMismatch();
  
  for (int i=0; i<A.rows*A.cols; i++)
    C.array[i] = F(A.array[i]);
}

double& Matrix::at(int row, int col) {
  if (trans) { // Swap
    int temp = col;
    col = row;
    row = temp;
  }
  if (row<0 || row>=rows || col<0 || col>=cols) 
    throw MatrixOutOfBounds(); // The indentation here doesn't work
  return array[cols*row + col];
}

double Matrix::at(int row, int col) const {
  if (trans) { // Swap
    int temp = col;
    col = row;
    row = temp;
  }
  if (row<0 || row>=rows || col<0 || col>=cols)
    throw MatrixOutOfBounds(); // The indentation here doesn't work
  return array[cols*row + col];
}

double Matrix::operator[](int index) {
  if (index<0 || index>=rows*cols)
    throw MatrixOutOfBounds();
  return array[index];
}

void Matrix::resize(int rows, int cols) {
  if (array) delete [] array;
  this->rows = rows; this->cols = cols;
  array = new double[rows*cols];
}

void Matrix::random(double max) {
  for (int i=0; i<rows*cols; i++) {
    array[i] = max*(2*drand48()-1);
  }
}

void Matrix::qrel() {
  array = 0;
}

void Matrix::qref(Matrix& M) {
  // Assumes that matrix M is of the same dimensions as this matrix
  if (array) delete [] array;
  array = M.array;
}

std::ostream& operator<<(std::ostream& out, const Matrix& M) {
  int width = M.getCols(), height = M.getRows();
  out << "{";
  for (int y=0; y<height; y++) {
    out << "{";
    for (int x=0; x<width; x++) {
      out << M.at(y,x);
      if (x!=width-1) out << ",";
    }
    out << "}";
    if (y!=height-1) out << ",";
  }
  out << "}";

  return out;
}

inline bool Matrix::checkDims(const Matrix& A, const Matrix& B) {
  return A.rows==B.rows && A.cols==B.cols;
}
