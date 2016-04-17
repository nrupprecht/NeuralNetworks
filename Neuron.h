/// Neuron.h - Header for Neuron class
/// Nathaniel Rupprecht 2016
///

#ifndef NEURONT_H
#define NEURONT_H

#include "Tensor.h"

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
  virtual void feedForward(const Tensor& input, Tensor& output, Tensor& Zout) = 0;
  virtual void backPropagate(const Tensor& deltaIn, Tensor& deltaOut, Tensor& Zout) = 0;
  virtual void updateDeltas(Tensor& aout, const Tensor& deltas) = 0; // aout not const so we can take the transpose
  virtual void gradientDescent(double factor) = 0;  
  virtual void clear() = 0;
  virtual void setTensor(int n, Tensor* M) = 0;
  virtual Tensor*& getTensor(int n) = 0;

  class OutOfBounds {};

 protected:
  vector<int> inShape, outShape;
};

class Sigmoid : public Neuron {
 public:
  Sigmoid(const vector<int>& inShape, const vector<int>& outShape, bool tr=false);
  ~Sigmoid();

  virtual void feedForward(const Tensor& input, Tensor& output, Tensor& Zout);
  virtual void backPropagate(const Tensor& input, Tensor& output, Tensor& Zout);
  virtual void updateDeltas(Tensor& aout, const Tensor& deltas);
  virtual void gradientDescent(double factor);
  virtual void clear();
  virtual void setTensor(int n, Tensor *M);
  virtual Tensor*& getTensor(int n);

  void setTransposed(bool t) { transposed = t; }
 protected:
  // Pointers to the matrices
  Tensor* weights;
  Tensor* biases;
  Tensor* wDeltas;
  Tensor* bDeltas;
  Tensor* diff;
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

  Tensor *wVelocity;
  Tensor *bVelocity;
};

#endif
