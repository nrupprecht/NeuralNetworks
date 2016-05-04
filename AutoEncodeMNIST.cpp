#include "Network.h"
#include "MNISTUnpack.h"

#include <sstream>
using std::stringstream;
#include <string>
using std::string;

int main(int argc, char* argv[]) {
  // Initialize MPI
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  
  // Get the training data
  FileUnpack unpacker("MNISTData/MNIST_trainingImages","MNISTData/MNIST_trainingLabels");
  unpacker.unpackInfo();
  auto inputs = unpacker.getImages();
  auto targets = unpacker.getLabels();

  if (rank==0) cout << "Autoencoder for MNIST" << endl;

  Network net;
  vector<int> neurons;

  neurons.push_back(784);
  neurons.push_back(200);

  net.setRate(0.01);
  net.setL2const(0.);

  net.createAutoEncoder(neurons, sigmoid, dsigmoid);

  net.setInputs(inputs);
  net.setTargets(inputs);

  net.setMinibatch(50);
  net.setTrainingIters(10);
  net.setDisplay(true);
  net.setCheckCorrect(false);
  
  net.setTrainingIters(10);
  
  // Write an initial image
  if (rank==0) {
    auto orig = *inputs.at(0);
    orig.reshape(28,28);  
    writeImage(orig, "Orig");
    cout << "Wrote Orig" << endl;
  }
  
  // Run network
  net.train();
  
  if (rank==0) {
    auto im = net.feedForward(*inputs.at(0));
    im.reshape(28,28);
    writeImage(im, "Image");
    cout << "Wrote Image" << endl;
  }  
  
  for (auto p : inputs) delete p;
  for (auto p : targets) delete p;
  
  // End MPI
  MPI_Finalize();

  return 0;
}
