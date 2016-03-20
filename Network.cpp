/// Network.cpp - Implements Network functions
/// Nathaniel Rupprecht 2016
///

#include "Network.h"

// Squaring function
inline double sqr(double x) { return x*x; }

Network::Network() : initialized(false), weights(0), biases(0), aout(0), zout(0), wDeltas(0), bDeltas(0), deltas(0), total(0), fnct(0), dfnct(0), rate(0.01), L2factor(0.01), trainingIters(100), minibatch(10), display(true) {};

Network::~Network() {
  if (weights) delete [] weights;
  if (biases) delete [] biases;
  if (aout) delete [] aout;
  if (zout) delete [] zout;
  if (wDeltas) delete [] wDeltas;
  if (bDeltas) delete [] bDeltas;
  if (deltas) delete [] deltas;
}

void Network::createFeedForward(vector<int> neurons, function F, function DF) {
  this->neurons = neurons;
  fnct = F; dfnct = DF;
  total = static_cast<int>(neurons.size());
  weights = new Matrix[total];
  biases = new Matrix[total];

  // Initialize vectors/matrices
  weights = new Matrix[total];
  biases = new Matrix[total];
  wDeltas = new Matrix[total];
  bDeltas = new Matrix[total];
  deltas = new Matrix[total];
  aout = new Matrix[total];
  zout = new Matrix[total];

  // Set vector/matrix sizes
  aout[0].resize(neurons.at(0), 1);
  zout[0].resize(neurons.at(0), 1);
  for (int i=1; i<total; i++) {
    weights[i].resize(neurons.at(i), neurons.at(i-1));
    biases[i].resize(neurons.at(i), 1);
    weights[i].random(1);
    biases[i].random(1);
    wDeltas[i].resize(neurons.at(i), neurons.at(i-1));
    bDeltas[i].resize(neurons.at(i), 1);
    deltas[i].resize(neurons.at(i), 1);
    aout[i].resize(neurons.at(i), 1);
    zout[i].resize(neurons.at(i), 1);
  }
  // The network has been initialized
  initialized = true;
}

void Network::train(vector<Matrix*>& inputs, vector<Matrix*>& targets, int subset) {
  if (inputs.size()!=targets.size()) return; // Mismatch
  if (!initialized) return; // Uninitialized

  if (subset<0) subset = inputs.size();

  int NData = subset;
  int nBatches = NData/minibatch;
  for (int iter=0; iter<trainingIters; iter++) {
    double aveError = 0;
    for (int i=0; i<nBatches; i++) {
      for (int j=0; j<minibatch; j++) {
	int index = i*minibatch+j;
	aout[0].qref(*inputs.at(index)); // Reference input
	feedForward();
	aveError += error(*targets.at(index));
	outputError(*targets.at(index));
	backPropagate();
	aout[0].qrel(); // Release reference
      }
      // Gradient descent
      gradientDescent();
    }
    // Iteration finished
    if (display) { // Display iteration summary
      cout << "Iteration " << iter << ":" << endl;
      cout << "Ave Error: " << aveError/inputs.size() << endl;
      // More later
    }
  }
  cout << "Training over." << endl;
}

Matrix Network::feedForward(Matrix& input) {
  aout[0].qref(input);
  for (int i=1; i<total; i++) {
    multiply(weights[i], aout[i-1], zout[i]); // W*a
    add(zout[i], biases[i], zout[i]); // + b
    apply(zout[i], fnct, aout[i]);
  }
  aout[0].qrel();
  return aout[total-1];
}

inline void Network::feedForward() {
  for (int i=1; i<total; i++) {
    multiply(weights[i], aout[i-1], zout[i]); // W*a
    add(zout[i], biases[i], zout[i]); // + b
    apply(zout[i], fnct, aout[i]);
  }
}

/// Squared error
inline double Network::error(const Matrix& target) {
  double error = 0;
  for (int i=0; i<target.getRows(); i++) 
    error += sqr(target.at(i,0) - aout[total-1].at(i,0));
  return error;
}

/// This error is the cross entropy
inline void Network::outputError(const Matrix& target) {
  subtract(aout[total-1], target, deltas[total-1]);
}

inline void Network::backPropagate() {
  for (int i=total-2; i>0; i--) {
    weights[i+1].T(); // Transpose
    Matrix acc(weights[i+1].getRows(), deltas[i+1].getCols());
    multiply(weights[i+1], deltas[i+1], acc);
    weights[i+1].T(); // Undo transpose
    apply(zout[i], dfnct, deltas[i]);
    hadamard(acc, deltas[i], deltas[i]);
  }
  for(int i=1; i<total; i++) {
      aout[i-1].T(); // Transpose
    Matrix diff(wDeltas[i].getRows(), wDeltas[i].getCols()); // Creating this every time may be to wasteful
    multiply(deltas[i], aout[i-1], diff);
    aout[i-1].T(); // Undo transpose
    add(wDeltas[i], diff, wDeltas[i]);
    add(bDeltas[i], deltas[i], bDeltas[i]);
  }
}

inline void Network::gradientDescent() {
  for (int i=1; i<total; i++) {
    multiply(rate, wDeltas[i], wDeltas[i]);
    multiply(rate, bDeltas[i], bDeltas[i]);
    subtract(weights[i], wDeltas[i], weights[i]);
    subtract(biases[i], bDeltas[i], biases[i]);
  }
}
