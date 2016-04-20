//  CIFARNet.cpp
/// Nathaniel Rupprecht 2016
///

#include "Network.h"
#include "CIFARUnpack.h"

int main(int argc, char* argv[]) {
  // Initialize MPI
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &size );

  CifarUnpacker unpacker;
  vector<string> fileNames;
  fileNames.push_back("CIFARData/data_batch_1.bin");
  fileNames.push_back("CIFARData/data_batch_2.bin");
  fileNames.push_back("CIFARData/data_batch_3.bin");
  fileNames.push_back("CIFARData/data_batch_4.bin");
  fileNames.push_back("CIFARData/data_batch_5.bin");
  unpacker.unpackInfo(fileNames);
  auto images = unpacker.getInputSet();
  auto labels = unpacker.getLabelSet();

  //unpacker.unpackInfo(vector<string>({string("CIFARData/data_batch_5.bin")}));
  //auto testImages = unpacker.getInputSet();
  //auto testLabels = unpacker.getLabelSet();

  Network net;
  vector<int> neurons;

  neurons.push_back(3072);
  neurons.push_back(500);
  neurons.push_back(100);
  neurons.push_back(10);
  
  net.setRate(0.001);
  net.setL2const(0.);
  net.createFeedForward(neurons, sigmoid, dsigmoid);

  if (rank==0) net.printDescription();

  net.setInputs(images);
  net.setTargets(labels);

  //net.setTestInputs(testImages);
  //net.setTestTargets(testLabels);

  net.setMinibatch(100);
  net.setTrainingIters(50);
  net.setCalcError(true);
  net.setDisplay(false);

  net.trainMPI();
  
  if (rank==0) {
    cout << "errRec=" << print(net.getErrorRec()) << ";\n";
    cout << "trainCorrect=" << print(net.getTrainPercentRec()) << ";\n";
    cout << "aveTime=" << net.getAveTime() << ";\n";
  }

  for (auto p : images) delete p;
  for (auto p : labels) delete p;
  //for (auto p : testImages) delete p;
  //for (auto p : testLabels) delete p;

  // End MPI
  MPI_Finalize();

  return 0;
}
