//

#include "MNISTUnpack.h"
#include "TestNet.h"

int main(int argc, char* argv[]) {
  Network net;

  // Get the training data
  FileUnpack unpacker("MNISTData/MNIST_trainingImages","MNISTData/MNIST_trainingLabels")\
    ;
  unpacker.unpackInfo();
  vector<Matrix*> inputs, targets;
  inputs = unpacker.getImages();
  targets = unpacker.getLabels();

  net.run(inputs, targets);
  return 0;
}
