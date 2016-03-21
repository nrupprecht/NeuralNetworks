/// Network.h - Header for Network class
/// Nathaniel Rupprecht 2016
///

#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
using std::vector;
#include <math.h>

#include "Matrix.h"

// The sigmoid function is a common function
inline double sigmoid(double x) {
  return 1.f/(1+exp(-x));
}

inline double dsigmoid(double x) {
  double sig = sigmoid(x);
  return sig*(1-sig);
}

class Layer {
 public:
  void feedForward(Matrix& input, Matrix& output);
  void backProp(Matrix&);

 private:
  
};

class Sigmoid : public Layer {
 public:

 private:
  
};

class Network {
 public:
  Network();
  ~Network();

  // Network initialization
  void createFeedForward(vector<int> neurons, function F, function DF);  

  // Network training/use
  void train(int subset=-1);
  Matrix feedForward(Matrix& input);

  // Mutators
  void setRate(double r) { rate = r; }
  void setL2Factor(double l2) { L2factor = l2; }
  void setTrainingIters(int i) { trainingIters = i; }
  void setMinibatch(int m) { minibatch = m; }
  void setDisplay(bool d) { display = d; }
  void setInputs(vector<Matrix*>& inputs) { this->inputs = inputs; }
  void setTargets(vector<Matrix*>& targets) { this->targets = targets; }
  void setTestInputs(vector<Matrix*>& inputs) { testInputs = inputs; }
  void setTestTargets(vector<Matrix*>& targets) { testTargets = targets; }

 private:
  // Network data
  function fnct, dfnct; // Neuron function and its derivative
  vector<int> neurons;
  int total;         // Total number of [a] arrays needed (number of layers including input)
  bool initialized;  // Whether a network has been initialized or not
  double rate;       // The learning rate
  double L2factor;   // The L2 norm factor
  int trainingIters; // The number of iterations we want to train for
  int minibatch;     // The minibatch size

  bool display; // Whether to display iteration data

  // Training/Testing data
  vector<Matrix*> inputs;
  vector<Matrix*> targets;
  bool doTest;
  vector<Matrix*> testInputs;
  vector<Matrix*> testTargets;

  // Neuron data
  Matrix *weights, *biases; // Network parameters
  Matrix *aout, *zout;      // For backpropagation
  // Deltas
  Matrix *wDeltas, *bDeltas, *deltas;

  // Helper functions
  inline void feedForward();
  inline bool checkMax(const Matrix& target);
  inline double error(const Matrix& target);
  inline void outputError(const Matrix& target);
  inline void backPropagate();
  inline void gradientDescent();
  inline void clearMatrices();
};

#endif
