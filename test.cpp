/// test.cpp - A file to test network functionality
/// Nathaniel Rupprecht 2016
///

#include "Network.h"
#include <iostream>
using std::cout;
using std::endl;

int main(int argc, char* argv[]) {
  Network net;

  vector<Matrix*> inputs, targets;
  Matrix M1(2,1), M2(2,1); // Keep these at a larger scope
  try {
    M1.at(0,0) = 1; M1.at(1,0) = 0;
    M2.at(0,0) = 0; M2.at(1,0) = 1;
    for (int i=0; i<100; i++) {
      if (i%2==0) {
	inputs.push_back(&M1);
	targets.push_back(&M1);
      }
      else {
	inputs.push_back(&M2);
	targets.push_back(&M2);
      }
    }
  }
  catch(...) {
    cout << "An error occured in seting up data.\n";
    return 1;
  }
  
  cout << "Data allocated" << endl;

  vector<int> neurons;
  neurons.push_back(2); neurons.push_back(2), neurons.push_back(2);
  net.createFeedForward(neurons, sigmoid, dsigmoid);

  cout << "Network created" << endl;

  try {
    net.setTrainingIters(500);
    net.setDisplay(false);
    net.train(inputs, targets);
  }
  catch(...) {
    cout << "An error occured in training.\n";
    return 2;
  }

  cout << net.feedForward(M1) << endl;
  cout << net.feedForward(M2) << endl;

  cout << "Training successful" << endl;

  return 0;
}
