/// MNISTNet.cpp
/// Nathaniel Rupprecht 2016
///

#include "Network.h"
#include "MNISTUnpack.h"

int main(int argc, char* argv[]) {
  // Initialize MPI
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &size );

  // Get the training data
  FileUnpack unpacker("MNISTData/MNIST_trainingImages","MNISTData/MNIST_trainingLabels");
  unpacker.unpackInfo();
  vector<Tensor*> inputs, targets;
  inputs = unpacker.getImages();
  targets = unpacker.getLabels();
  
  FileUnpack testUnpacker("MNISTData/MNIST_testImages", "MNISTData/MNIST_testLabels");
  testUnpacker.unpackInfo();
  vector<Tensor*> testInputs, testTargets;
  testInputs = testUnpacker.getImages();
  testTargets = testUnpacker.getLabels();

  Network net;
  vector<int> neurons;
  
  neurons.push_back(784);
  neurons.push_back(500);
  neurons.push_back(30);
  neurons.push_back(10);
  
  net.setRate(0.1);
  net.setL2const(0.);  
  net.createFeedForward(neurons, sigmoid, dsigmoid);

  net.setInputs(inputs);
  net.setTargets(targets);

  net.setTestInputs(testInputs);
  net.setTestTargets(testTargets);

  net.setMinibatch(50);
  net.setTrainingIters(50);
  net.setCalcError(true);
  net.setDisplay(false);

  if (rank==0) net.printDescription();
  net.trainMPI();

  if (rank==0) {
    cout << "errRec=" << print(net.getErrorRec()) << ";\n";
    cout << "trainCorrect=" << print(net.getTrainPercentRec()) << ";\n";
    cout << "aveTime=" << net.getAveTime() << ";\n";
    cout << "errVtime=" << print(net.getErrVTime()) << ";\n";
  }
  
  for (auto p : inputs) delete p;
  for (auto p : targets) delete p;
  for (auto p : testInputs) delete p;
  for (auto p : testTargets) delete p;  

  // End MPI
  MPI_Finalize();

  return 0;
}
