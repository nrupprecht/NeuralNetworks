#ifndef TESTNET_H
#define TESTNET_H

#include <vector>
using std::vector;

#include "Neuron.h"

template<typename T> T sqr(T x) { return x*x; }

class Network {
 public:
  Network();
  ~Network();

  void run(vector<Matrix*> inputs, vector<Matrix*> targets);

 private:

  inline double error(const Matrix& target);
  inline bool checkMax(const Matrix& target);

  int total;

  Matrix* aout;
  Matrix* zout;
  Matrix* deltas;

  Neuron** layers;

};

#endif 
