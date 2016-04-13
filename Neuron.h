/// Neuron.h - Header for Neuron class
/// Nathaniel Rupprecht 2016
///

#ifndef NEURON_H
#define NEURON_H

#include "Matrix.h"

#include <vector>
using std::vector;

// The sigmoid function is a common function
inline double sigmoid(double x) {
  return 1.f/(1+exp(-x));
}

inline double dsigmoid(double x) {
  double sig = sigmoid(x);
  return sig*(1-sig);
}

class Neuron {
 public:
  Neuron(const vector<int>& inShape, const vector<int>& outShape);
  virtual void feedForward(const Matrix& input, Matrix& output, Matrix& Zout) = 0;
  virtual void backPropagate(const Matrix& deltaIn, Matrix& deltaOut, Matrix& Zout) = 0;
  virtual void updateDeltas(Matrix& aout, const Matrix& deltas) = 0; // aout not const so we can take the transpose
  virtual void gradientDescent(double factor) = 0;  
  virtual void clear() = 0;
  virtual void setMatrix(int n, Matrix* M) = 0;

  class OutOfBounds {};

 protected:
  vector<int> inShape, outShape;
};

class Sigmoid : public Neuron {
 public:
  Sigmoid(const vector<int>& inShape, const vector<int>& outShape, bool tr=false);
  ~Sigmoid();

  virtual void feedForward(const Matrix& input, Matrix& output, Matrix& Zout);
  virtual void backPropagate(const Matrix& input, Matrix& output, Matrix& Zout);
  virtual void updateDeltas(Matrix& aout, const Matrix& deltas);
  virtual void gradientDescent(double factor);
  virtual void clear();
  virtual void setMatrix(int n, Matrix *M);

 protected:
  // Pointers to the matrices
  Matrix* weights;
  Matrix* biases;
  Matrix* wDeltas;
  Matrix* bDeltas;
  Matrix* diff;
  bool owned;
  bool transposed;

  double L2factor;

  // Input and output shapes
  vector<int> inShape;
  vector<int> outShape;

  // Activation function and its derivative
  double (*fnct) (double);
  double (*dfnct) (double);
};

class SigmoidM : public Sigmoid {
 public:
  SigmoidM(const vector<int>& inShape, const vector<int>& outShape, bool tr=false);
  ~SigmoidM();
  
  virtual void gradientDescent(double factor);
  
 private:
  // Pointers to the matrices
  double decay;

  Matrix *wVelocity;
  Matrix *bVelocity;
};

#endif
