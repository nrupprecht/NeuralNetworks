/// MNISTNet.cpp
/// Nathaniel Rupprecht 2016
///

#include "Network.h"
#include "MNISTUnpack.h"

int main(int argc, char* argv[]) {
  
  // Get the training data
  FileUnpack unpacker("MNISTData/MNIST_trainingImages","MNISTData/MNIST_trainingLabels");
  unpacker.unpackInfo();
  vector<Matrix*> inputs, targets;
  inputs = unpacker.getImages();
  targets = unpacker.getLabels();
  
  FileUnpack testUnpacker("MNISTData/MNIST_testImages", "MNISTData/MNIST_testLabels");
  testUnpacker.unpackInfo();
  vector<Matrix*> testInputs, testTargets;
  testInputs = testUnpacker.getImages();
  testTargets = testUnpacker.getLabels();

  Network net;
  vector<int> neurons;
  
  neurons.push_back(784);
  neurons.push_back(150); 
  neurons.push_back(50);
  neurons.push_back(10);

  net.setRate(0.001);
  
  net.createFeedForward(neurons, sigmoid, dsigmoid);

  net.setInputs(inputs);
  net.setTargets(targets);

  net.setTestInputs(testInputs);
  net.setTestTargets(testTargets);

  net.setTrainingIters(10);
  net.setDisplay(true);
  net.train();

  for (auto p : inputs) delete p;
  for (auto p : targets) delete p;
  for (auto p : testInputs) delete p;
  for (auto p : testTargets) delete p;
  
  return 0;
}
