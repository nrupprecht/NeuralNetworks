#include "Neuron.h"

// For debugging
#include <iostream>
using std::cout;
using std::endl;

Neuron::Neuron(const vector<int>& inShape, const vector<int>& outShape) : inShape(inShape), outShape(outShape) {};

Sigmoid::Sigmoid(const vector<int>& inShape, const vector<int>& outShape, bool tr) : Neuron(inShape, outShape), L2factor(0) {
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
  transposed = tr;
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
  if (transposed) weights->T();
  multiply(*weights, input, Zout);
  if (transposed) weights->T();
  NTplusEqUnsafe(Zout, *biases);
  apply(Zout, fnct, output);
}

void Sigmoid::backPropagate(const Matrix& deltaIn, Matrix& deltaOut, Matrix& Zout) {
  if (!transposed) weights->T();
  Matrix acc(weights->getRows(), deltaIn.getCols());
  multiply(*weights, deltaIn, acc);
  if (!transposed) weights->T();  
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

  if (transposed) {
    weights->T();
    subtract(*weights, *wDeltas, *weights);
    weights->T();
  }
  else NTminusEqUnsafe(*weights, *wDeltas);
  NTminusEqUnsafe(*biases, *bDeltas);
}

inline void Sigmoid::clear() {
  wDeltas->zero();
  bDeltas->zero();
}

void Sigmoid::setMatrix(int n, Matrix* M) {
  // Do the naieve thing for now

  switch (n) {
  case 0: {
    weights = M;
    break;
  }
  case 1: {
    biases = M;
    break;
  }
  case 2: {
    wDeltas = M;
    break;
  }
  case 3: {
    bDeltas = M;
    break;
  }
  default: throw OutOfBounds();
  }
}

SigmoidM::SigmoidM(const vector<int>& inShape, const vector<int>& outShape, bool tr) : Sigmoid(inShape, outShape, tr), decay(0.001) {
  int in = inShape.at(0), out = outShape.at(0);
  wVelocity = new Matrix(out, in);
  bVelocity = new Matrix(out, 1);
}

SigmoidM::~SigmoidM() {
  Sigmoid::~Sigmoid();
  if (wVelocity) delete wVelocity;
  if (bVelocity) delete bVelocity;
}

void SigmoidM::gradientDescent(double factor) {
  double mult = 1-L2factor/weights->getCols();
  timesEq(*weights, mult); // L2 normalization
  multiply(factor, *wDeltas, *wDeltas);
  multiply(factor, *bDeltas, *bDeltas);

  // Velocity decay
  timesEq(*wVelocity, 1-decay);
  timesEq(*bVelocity, 1-decay);

  timesEq(*wDeltas, factor);
  timesEq(*bDeltas, factor);
  // Update velocity
  NTplusEqUnsafe(*wVelocity, *wDeltas);
  NTplusEqUnsafe(*bVelocity, *bDeltas);
  
  if (transposed) {
    weights->T();
    subtract(*weights, *wVelocity, *weights);
    weights->T();
  }
  else NTminusEqUnsafe(*weights, *wVelocity);
  NTminusEqUnsafe(*biases, *bVelocity); 
}
