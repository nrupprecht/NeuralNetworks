#include "Neuron.h"

// For debugging
#include <iostream>
using std::cout;
using std::endl;

Neuron::Neuron(const vector<int>& inShape, const vector<int>& outShape) : inShape(inShape), outShape(outShape) {};

Sigmoid::Sigmoid(const vector<int>& inShape, const vector<int>& outShape, bool tr) : Neuron(inShape, outShape), L2factor(0) {
  // Assume the input/output is a vector (n, 1)
  int in = inShape.at(0), out = outShape.at(0);
  weights = new Tensor(out, in);
  weights->random(1/sqrt(in));
  wDeltas = new Tensor(out, in);
  biases = new Tensor(out, 1);
  biases->random();
  bDeltas = new Tensor(out, 1);
  diff = new Tensor(out, in);

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

void Sigmoid::feedForward(const Tensor& input, Tensor& output, Tensor& Zout) {
  int aI = 1;
  if (transposed) aI = 0;
  multiply(*weights, aI, input, 0, Zout);
  NTplusEqUnsafe(Zout, *biases);
  apply(Zout, fnct, output);
}

void Sigmoid::backPropagate(const Tensor& deltaIn, Tensor& deltaOut, Tensor& Zout) {
  int aI = 0, d = weights->getCols(); // getRows()
  if (transposed) {
    aI = 1;
    d = weights->getRows();
  }
  Tensor acc(d, deltaIn.getCols());
  multiply(*weights, aI, deltaIn, 0, acc);
  apply(Zout, dfnct, deltaOut);
  hadamard(acc, deltaOut, deltaOut);
}

void Sigmoid::updateDeltas(Tensor& Aout, const Tensor& deltas) {
  multiply(deltas, 1, Aout, 1, *diff);
  NTplusEqUnsafe(*wDeltas, *diff);
  NTplusEqUnsafe(*bDeltas, deltas);
}

void Sigmoid::gradientDescent(double factor) {
  double mult = 1-L2factor/weights->getCols();
  timesEq(*weights, mult); // L2 normalization

  timesEq(*wDeltas, factor);
  //multiply(factor, *wDeltas, *wDeltas);
  timesEq(*bDeltas, factor);
  //multiply(factor, *bDeltas, *bDeltas);

  if (transposed) {
    //weights->T();
    subtract(*weights, *wDeltas, *weights); //** Problem here
    //weights->T();
  }
  else NTminusEqUnsafe(*weights, *wDeltas);
  NTminusEqUnsafe(*biases, *bDeltas);
}

inline void Sigmoid::clear() {
  wDeltas->zero();
  bDeltas->zero();
}

void Sigmoid::setTensor(int n, Tensor* M) {
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

Tensor*& Sigmoid::getTensor(int n) {
  switch (n) {
  case 0: return weights;
  case 1: return biases;
  case 2: return wDeltas;
  case 3: return bDeltas;
  default: throw OutOfBounds();
  }
}

SigmoidM::SigmoidM(const vector<int>& inShape, const vector<int>& outShape, bool tr) : Sigmoid(inShape, outShape, tr), decay(0.001) {
  int in = inShape.at(0), out = outShape.at(0);
  wVelocity = new Tensor(out, in);
  bVelocity = new Tensor(out, 1);
}

SigmoidM::~SigmoidM() {
  Sigmoid::~Sigmoid();
  if (wVelocity) delete wVelocity;
  if (bVelocity) delete bVelocity;
}

void SigmoidM::gradientDescent(double factor) {
  double mult = 1-L2factor/weights->getCols();
  timesEq(*weights, mult); // L2 normalization

  timesEq(*wDeltas, factor);
  //multiply(factor, *wDeltas, *wDeltas);
  timesEq(*bDeltas, factor);
  //multiply(factor, *bDeltas, *bDeltas);

  // Velocity decay
  timesEq(*wVelocity, 1-decay);
  timesEq(*bVelocity, 1-decay);

  timesEq(*wDeltas, factor);
  timesEq(*bDeltas, factor);
  // Update velocity
  NTplusEqUnsafe(*wVelocity, *wDeltas);
  NTplusEqUnsafe(*bVelocity, *bDeltas);
  
  if (transposed) {
    //weights->T();
    subtract(*weights, *wVelocity, *weights); //** PROBLEM
    //weights->T();
  }
  else NTminusEqUnsafe(*weights, *wVelocity);
  NTminusEqUnsafe(*biases, *bVelocity); 
}
