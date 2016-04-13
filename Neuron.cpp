#include "Neuron.h"

// For debugging
#include <iostream>
using std::cout;
using std::endl;

Neuron::Neuron(const vector<int>& inShape, const vector<int>& outShape) : inShape(inShape), outShape(outShape) {};

Sigmoid::Sigmoid(const vector<int>& inShape, const vector<int>& outShape) : Neuron(inShape, outShape), L2factor(0) {
  // Assume the input/output is a vector (n, 1)
  int in = inShape.at(0), out = outShape.at(0);
  weights = new Matrix(out, in);
  weights->random(1/sqrt(in));
  wDeltas = new Matrix(out, in);
  biases = new Matrix(out, 1);
  biases->random();
  bDeltas = new Matrix(out, 1);
  diff = new Matrix(out, in);

  fnct = sigmoid;
  dfnct = dsigmoid;

  owned = true;
}

Sigmoid::~Sigmoid() {
  if (owned) {
    if (weights) delete [] weights;
    if (biases) delete [] biases;
    if (wDeltas) delete [] wDeltas;
    if (bDeltas) delete [] bDeltas;
  }
  if (diff) delete [] diff;
}

void Sigmoid::feedForward(const Matrix& input, Matrix& output, Matrix& Zout) {
  multiply(*weights, input, Zout);
  NTplusEqUnsafe(Zout, *biases);
  apply(Zout, fnct, output);
}

void Sigmoid::backPropagate(const Matrix& deltaIn, Matrix& deltaOut, Matrix& Zout) {
  weights->T();
  Matrix acc(weights->getRows(), deltaIn.getCols());
  multiply(*weights, deltaIn, acc);
  weights->T();  
  apply(Zout, dfnct, deltaOut);
  hadamard(acc, deltaOut, deltaOut);
}

void Sigmoid::updateDeltas(Matrix& Aout, const Matrix& deltas) {
  Aout.T();
  multiply(deltas, Aout, *diff);
  Aout.T();

  NTplusEqUnsafe(*wDeltas, *diff);
  NTplusEqUnsafe(*bDeltas, deltas);
}

void Sigmoid::gradientDescent(double factor) {
  double mult = 1-L2factor/weights->getCols();
  timesEq(*weights, mult); // L2 normalization

  multiply(factor, *wDeltas, *wDeltas);
  multiply(factor, *bDeltas, *bDeltas);

  NTminusEqUnsafe(*weights, *wDeltas);
  NTminusEqUnsafe(*biases, *bDeltas);
}

inline void Sigmoid::clear() {
  wDeltas->zero();
  bDeltas->zero();
}
