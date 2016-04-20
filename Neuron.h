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
  Neuron(const Shape& inShape, const Shape& outShape);
  virtual void feedForward(const Tensor& input, Tensor& output, Tensor& Zout) = 0;
  virtual void backPropagate(const Tensor& deltaIn, Tensor& deltaOut, Tensor& Zout) = 0;
  virtual void updateDeltas(Tensor& aout, const Tensor& deltas) = 0; // aout not const so we can take the transpose
  virtual void gradientDescent(double factor) = 0;  
  virtual void clear() = 0;
  virtual void setTensor(int n, Tensor* M) = 0;
  virtual Tensor*& getTensor(int n) = 0;
  virtual vector<Tensor*> getCommon() = 0;

  class OutOfBounds {};

  Shape getInShape() const { return inShape; }
  Shape getOutShape() const { return outShape; }

 protected:
  Shape inShape, outShape;
};

class Sigmoid : public Neuron {
 public:
  Sigmoid(const Shape& inShape, const Shape& outShape, bool tr=false);
  ~Sigmoid();

  virtual void feedForward(const Tensor& input, Tensor& output, Tensor& Zout);
  virtual void backPropagate(const Tensor& input, Tensor& output, Tensor& Zout);
  virtual void updateDeltas(Tensor& aout, const Tensor& deltas);
  virtual void gradientDescent(double factor);
  virtual void clear();
  virtual void setTensor(int n, Tensor *M);
  virtual Tensor*& getTensor(int n);
  virtual vector<Tensor*> getCommon();

  void setTransposed(bool t) { transposed = t; }
 protected:
  // Pointers to the matrices
  Tensor* weights;
  Tensor* biases;
  Tensor* wDeltas;
  Tensor* bDeltas;
  Tensor* diff;
  Tensor acc;
  bool owned;
  bool transposed;

  double L2factor;

  // Input and output shapes
  Shape inShape;
  Shape outShape;

  // Activation function and its derivative
  double (*fnct) (double);
  double (*dfnct) (double);
};

#endif
