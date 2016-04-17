///  CIFARNet.cpp
/// Nathaniel Rupprecht 2016
///

#include "Network.h"
#include "CIFARUnpack.h"

int main(int argc, char* argv[]) {

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

  unpacker.unpackInfo(vector<string>({string("CIFARData/data_batch_5.bin")}));
  auto testImages = unpacker.getInputSet();
  auto testLabels = unpacker.getLabelSet();

  Network net;
  vector<int> neurons;

  neurons.push_back(3072);
  neurons.push_back(500);
  neurons.push_back(10);

  net.setRate(0.001);
  net.setMinibatch(10);
  net.createFeedForward(neurons, sigmoid, dsigmoid);

  net.setInputs(images);
  net.setTargets(labels);

  //net.setTestInputs(testImages);
  //net.setTestTargets(testLabels);

  net.setTrainingIters(50);
  net.setDisplay(true);
  net.train();
  
}
