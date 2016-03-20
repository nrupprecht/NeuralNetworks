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

  Network net;
  vector<int> neurons;

  neurons.push_back(784);
  neurons.push_back(50); 
  neurons.push_back(10);
  
  net.createFeedForward(neurons, sigmoid, dsigmoid);

  net.setTrainingIters(100);
  net.setDisplay(true);
  net.train(inputs, targets);
  
  return 0;
}
