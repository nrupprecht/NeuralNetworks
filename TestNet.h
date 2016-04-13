#ifndef TESTNET_H
#define TESTNET_H

#include <vector>
using std::vector;

#include "Neuron.h"

class Network {
 public:
  Network();
  ~Network();

  void run(vector<Matrix*> inputs, vector<Matrix*> targets);

 private:

  inline bool checkMax(const Matrix& target);

  int total;

  Matrix* aout;
  Matrix* zout;
  Matrix* deltas;

  Neuron** layers;

};

#endif 
