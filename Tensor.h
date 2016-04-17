/// Tensor.h
/// Nathaniel Rupprecht 2016
///

#ifndef TENSOR_H
#define TENSOR_H

#include "Utility.h"
#include "Shape.h"

/// Index class
struct Index {
  Index(int I) : num(true), index(I) {};
  Index(char c) : num(false), index(c) {};

  bool num;
  int index;
};

/// Tensor class
class Tensor {
 public:
 Tensor() : array(0), total(0), stride(0), shape(Shape()) {};
  Tensor(Shape s);
  template<typename ...T> Tensor(int first, T... last) {
    Shape s(first, last...);
    initialize(s);
  }
  Tensor(const Tensor& T);
  ~Tensor();

  Tensor& operator=(const Tensor& T);
  
  // "at" function
  double& at(uint i) {
    if (shape.dims[0]<=i) throw TensorDimsMismatch();
    return array[i*stride[0]];
  }
  template<typename ...T> double& at(uint first, T ...s) {
    int address = 0;
    at_address(address, 0, first, s...);    
    return array[address];
  }

  double at(uint i) const { 
    if (shape.dims[0]>=i) throw TensorDimsMismatch();
    return array[i*stride[0]]; 
  }
  template<typename ...T> double at(uint first, T ...s) const {
    int address = 0;
    at_address(address, 0, first, s...);    
    return array[address];
  }

  double& at(vector<int> indices) {
    if (indices.size()>shape.rank) throw TensorRankMismatch();
    int add = 0;
    for (int i=0; i<shape.rank; i++)
      add += stride[i]*indices.at(i);
    return array[add];
  }

  double at(vector<int> indices) const {
    if (indices.size()>shape.rank) throw TensorRankMismatch();
    int add = 0;
    for (int i=0; i<shape.rank; i++)
      add += stride[i]*indices.at(i);
    return array[add];
  }

  /// Special (matrix type) accessors
  int getRows() const {
    int rank = shape.rank;
    if (rank<2) return shape.dims[rank-1]; // For a pure vector
    return shape.dims[rank-2];
  }
  int getCols() const { return shape.dims[shape.rank-1]; }

  /// Arithmetic functions
  friend void multiply(const Tensor& A, int aI, const Tensor& B, int bI, Tensor& C);
  friend void multiply(const Tensor& A, const Tensor& B, Tensor& C);
  friend void multiply(const double m, const Tensor& A, const Tensor& B);
  friend void timesEq(Tensor& A, const double m);
  friend void add(const Tensor& A, const Tensor& B, Tensor& C);
  friend void NTplusEqUnsafe(Tensor& A, const Tensor& B, double mult=1.);
  friend void subtract(const Tensor&A, const Tensor& B, Tensor& C);
  friend void NTminusEqUnsafe(Tensor& A, const Tensor& B, double mult=1.);
  friend void hadamard(const Tensor&A, const Tensor& B, Tensor& C);
  friend void apply(const Tensor& A, function F, Tensor& C);

  /// Accessors
  int size() const { return total; }
  int getRank() { return shape.rank; }
  int getDim(int i);
  Shape getShape() const { return shape; }

  // Setting Tensors
  template<typename ...T> void resize(int first, T... last) {
    if (array) delete [] array;
    if (stride) delete [] stride;
    *this = Tensor(first, last...);
  };
  template<typename ...T> void reshape(int first, T... last) {
    Shape s(first, last...);
    int tot = s.getTotal();
    if (total!=tot) throw TensorBadReshape();
    shape = s;
    total = tot;
  };
  void random(double max=1); // Written
  void zero();
  
  /// Quick handling of tensors
  void qrel();          // Release array memory
  void qref(Tensor& T); // Reference this tensor's array

  /// Error classes
  class TensorOutOfBounds {};
  class TensorRankMismatch {};
  class TensorDimsMismatch {};
  class TensorBadReshape {};
  class TensorBadContraction {};

  /// Printing and reading
  friend std::ostream& operator<<(std::ostream& out, const Tensor& T);

  // private: //**
  /// Helper functions
  void initialize(Shape s);
  template<typename ...T> void at_address(int&, int) const {};
  template<typename ...T> void at_address(int& add, int step, int first, T ... last) const {
    if (step>=shape.rank || first>=shape.dims[step]) throw TensorOutOfBounds();
    add += stride[step]*first;
    at_address(add, step+1, last...);
  }

  static inline bool checkDims(const Tensor& A, const Tensor& B);
  static inline void writeHelper(vector<int> indices, std::ostream& out, const Tensor& T);
  
  
  /// Data
  Shape shape; // The shape of the tensor
  int *stride; // The stride for each dimension
  int total;   // The total number of entries
  double *array; // The entries of the tensor
};

#endif
