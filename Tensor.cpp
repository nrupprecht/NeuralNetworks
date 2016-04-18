#include "Tensor.h"

Tensor::Tensor(Shape s) {
  initialize(s);
}

Tensor::Tensor(const Tensor& T) {
  initialize(T.shape);
  for (int i=0; i<total; i++) array[i] = T.array[i];
}

Tensor& Tensor::operator=(const Tensor& T) {
  if (array) delete [] array;
  if (stride) delete [] stride;
  
  initialize(T.shape);
  // Set values
  for (int i=0; i<total; i++) array[i] = T.array[i];
}

Tensor::~Tensor() {
  if (array) delete [] array;
  if (stride) delete [] stride;
}

void multiply(const Tensor& A, int aI, const Tensor& B, int bI, Tensor& C) {
  // Check index correctness
  if (aI>=A.shape.rank || bI>=B.shape.rank) throw Tensor::TensorBadContraction();
  // Check Matrix compatability
  if (A.shape.rank + B.shape.rank - 2 != C.shape.rank) throw Tensor::TensorRankMismatch();

  int i, j;
  for (i=0, j=0; i<A.shape.rank; i++) {
    if (i==aI) continue;
    if (A.shape.dims[i]!=C.shape.dims[j]) throw Tensor::TensorDimsMismatch();
    j++;
  }
  for (i=0; i<B.shape.rank; i++) {
    if (i==bI) continue;
    if (B.shape.dims[i]!=C.shape.dims[j]) throw Tensor::TensorDimsMismatch();
    j++;
  }

  // Special case - (m,k) times (k,n) -> (m, n)
  if (A.shape.rank==2 && B.shape.rank==2) { // Normal matrix multiplication
    int m, n, k, ac, bc, cc;

    auto AT = CblasTrans;
    if (aI==0) { // A is "transposed"
      m = A.shape.dims[1];
      k = A.shape.dims[0];
    }
    else {
      m = A.shape.dims[0];
      k = A.shape.dims[1];
      AT = CblasNoTrans;
    }
    ac = A.shape.dims[1];

    auto BT = CblasNoTrans;
    if (bI==0) n = B.shape.dims[1]; // B not "transposed"
    else {
      n = B.shape.dims[0];
      BT = CblasTrans;
    }
    bc = B.shape.dims[1];
    cc = C.shape.dims[1];
    double ALPHA = 1.0, BETA = 0;

    cblas_dgemm(CblasRowMajor, AT, BT, m, n, k, ALPHA, A.array, ac, B.array, bc, BETA, C.array, cc);
    return;
  }
  if (A.shape.rank==2 && B.shape.rank==1) { // Matrix times vector (k)->(k,1)
    int m, n, k, ac, bc, cc;
    auto AT = CblasTrans;
    if (aI==0) { // A is "transposed"
      m = A.shape.dims[1];
      k = A.shape.dims[0];
    }
    else {
      m = A.shape.dims[0];
      k = A.shape.dims[1];
      AT = CblasNoTrans;
    }
    ac = A.shape.dims[1];
    auto BT = CblasNoTrans;
    n = bc = cc = 1;
    double ALPHA = 1.0, BETA = 0;

    cblas_dgemm(CblasRowMajor, AT, BT, m, n, k, ALPHA, A.array, ac, B.array, bc, BETA, C.array, cc);
    return;
  }

  // STUB
  
}

void multiply(const Tensor& A, const Tensor& B, Tensor& C) {
  multiply(A, A.shape.rank-1, B, 0, C);
}

void multiply(const double m, const Tensor& A, const Tensor& B) {
  for (int i=0; i<A.total; i++) B.array[i] = m*A.array[i];
}

void timesEq(Tensor& A, const double m) {
  for (int i=0; i<A.total; i++) A.array[i] *= m;
}

void add(const Tensor &A, const Tensor& B, Tensor& C) {
  Tensor::checkDims(A, B); Tensor::checkDims(A, C);
  for (int i=0; i<A.total; i++) C.array[i] = A.array[i] + B.array[i];
}

void NTplusEqUnsafe(Tensor& A, const Tensor& B, double mult) {
  for (int i=0; i<A.total; i++) A.array[i] += mult*B.array[i];
}

void subtract(const Tensor &A, const Tensor& B, Tensor& C) {
  Tensor::checkDims(A, B); Tensor::checkDims(A, C);
  for (int i=0; i<A.total; i++) C.array[i] = A.array[i] - B.array[i];
}

void NTminusEqUnsafe(Tensor& A, const Tensor& B, double mult) {
  for (int i=0;i<A.total; i++) A.array[i] -= mult*B.array[i];
}

void hadamard(const Tensor &A, const Tensor& B, Tensor& C) {
  Tensor::checkDims(A, B); Tensor::checkDims(A, C);
  for (int i=0; i<A.total; i++) C.array[i] = A.array[i] * B.array[i];
}

void apply(const Tensor& A, function F, Tensor& C) {
  Tensor::checkDims(A, C);
  for (int i=0; i<A.total; i++) C.array[i] = F(A.array[i]);
}

int Tensor::getDim(int i) {
  if (i<0 || i>shape.rank) throw TensorRankMismatch();
  return shape.dims[i];
}

void Tensor::random(double max) {
  for (int i=0; i<total; i++)
    array[i] = max*(2*drand48()-1);
}

void Tensor::zero() {
  for (int i=0; i<total; i++) array[i] = 0;
}

void Tensor::qrel() {
  array = 0;
}

void Tensor::qref(Tensor& T) {
  if (array) delete [] array;
  array = T.array;
}

inline void Tensor::writeHelper(vector<int> indices, std::ostream& out, const Tensor& T) {
  out << '{';
  int step = indices.size();
  if (step==T.shape.rank-1) { // Base case
    for (int i=0; i<T.shape.dims[step]; i++) {
      indices.push_back(i); // Add i
      out << T.at(indices);
      indices.pop_back();   // Remove i
      if (i!=T.shape.dims[step]-1) out << ',';
    }
  }
  else {
    for (int i=0; i<T.shape.dims[step]; i++) {
      indices.push_back(i); // Add i
      writeHelper(indices, out, T);
      indices.pop_back();   // Remove i
      if (i!=T.shape.dims[step]-1) out << ',';
    }
  }
  out << '}';
}

std::ostream& operator<<(std::ostream& out, const Tensor& T) {
  vector<int> indices;
  Tensor::writeHelper(indices, out, T);
  return out;
}

void Tensor::initialize(Shape s, bool del) {
  shape = s;

  // Find total
  total = 1;
  for (int i=0; i<shape.rank; i++) total *= shape.dims[i];
  
  // Set stride array
  int count = 1;
  stride = new int[shape.rank];
  for (int i=0; i<shape.rank; i++) {
    count *= shape.dims[i];
    stride[i] = total/count;
  }

  if (del) {
    // Set data array
    array = new double[total];
    for (int i=0; i<total; i++) array[i] = 0.;
  }
}

inline bool Tensor::checkDims(const Tensor& A, const Tensor& B) {
  if (A.shape.rank != B.shape.rank) throw TensorRankMismatch();
  for (int i=0; i<A.shape.rank; i++) 
    if (A.shape.dims[i]!=B.shape.dims[i])
      throw TensorDimsMismatch();
}
