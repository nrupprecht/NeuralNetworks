#include "Network.h"
#include "MNISTUnpack.h"

#include <sstream>
using std::stringstream;
#include <string>
using std::string;

int main(int argc, char* argv[]) {
  // Get the training data
  FileUnpack unpacker("MNISTData/MNIST_trainingImages","MNISTData/MNIST_trainingLabels");
  unpacker.unpackInfo();
  auto inputs = unpacker.getImages();
  auto targets = unpacker.getLabels();

  cout << "Autoencoder for MNIST" << endl;

  Network net;
  vector<int> neurons;

  neurons.push_back(784);
  neurons.push_back(100);

  net.setRate(0.01);
  net.setL2const(0.);

  net.createAutoEncoder(neurons, sigmoid, dsigmoid);

  net.setInputs(inputs);
  net.setTargets(inputs);

  net.setMinibatch(10);
  net.setTrainingIters(10);
  net.setDisplay(true);
  net.setCheckCorrect(false);
  
  net.setTrainingIters(50);
  
  // Write an initial image
  auto orig = *inputs.at(0);
  orig.reshape(28,28);

  auto s_orig = orig.shift(Shape(5,5));
  writeImage(s_orig, "shifted");

  writeImage(orig, "Orig");
  cout << "Wrote Orig" << endl;
  
  auto im = net.feedForward(*inputs.at(0));
  im.reshape(28,28);
  writeImage(im, "Image", 0);
  cout << "Wrote Image0" << endl;
  
  // Run network

  net.train();
  
  im = net.feedForward(*inputs.at(0));
  im.reshape(28,28);
  writeImage(im, "Image");
  cout << "Wrote Image" << endl;

  for (auto p : inputs) delete p;
  for (auto p : targets) delete p;

  return 0;
}
