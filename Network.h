/// Network.h - Header for Network class
/// Nathaniel Rupprecht 2016
///

#ifndef NETWORK_H
#define NETWORK_H

#include "Neuron.h"
#include "EasyBMP/EasyBMP.h"

inline void createImage(Tensor& M, BMP& image) {
  int width = M.getCols(), height = M.getRows();
  image.SetSize(width, height);
  for (int i=0; i<height; i++)
    for (int j=0; j<width; j++) {
      int col = 255*M.at(i,j);
      image.SetPixel(j, i, RGBApixel(col, col, col));
    }
}

inline void writeImage(Tensor& M, string fileName, int mark=-1) {
  BMP image;
  if (mark>-1) {
    stringstream stream;
    stream << fileName << mark;
    fileName.clear();
    stream >> fileName;
  }
  fileName += ".bmp";
  createImage(M, image);
  image.WriteToFile(fileName.c_str());
}

/// The Network class
class Network {
 public:
  Network();
  ~Network();

  // Network initialization
  void createFeedForward(vector<int>& neurons, function F, function DF);  
  void createAutoEncoder(vector<int>& neurons, function F, function DF);

  // Network training/use
  void train(int subset=-1);
  Tensor feedForward(Tensor& input);

  // Mutators
  void setRate(double r) { rate = r; }
  void setL2const(double l2) { L2const = l2; }
  void setTrainingIters(int i) { trainingIters = i; }
  void setMinibatch(int m) { minibatch = m; }
  void setDisplay(bool d) { display = d; }
  void setCheckCorrect(bool c) { checkCorrect = c; }
  void setInputs(vector<Tensor*>& inputs) { this->inputs = inputs; }
  void setTargets(vector<Tensor*>& targets) { this->targets = targets; }
  void setTestInputs(vector<Tensor*>& inputs) { testInputs = inputs; }
  void setTestTargets(vector<Tensor*>& targets) { testTargets = targets; }

 private:
  // Network data
  function fnct, dfnct; // Neuron function and its derivative
  vector<int> neurons;
  int total;         // Total number of [a] arrays needed (number of layers including input)
  bool initialized;  // Whether a network has been initialized or not
  double rate;       // The learning rate
  double factor;     // rate / minibatch
  double L2const;    // The L2 penalty
  double L2factor;   // L2const * rate
  int trainingIters; // The number of iterations we want to train for
  int minibatch;     // The minibatch size

  bool display; // Whether to display iteration data
  bool checkCorrect; // Whether to check whether training data was correct

  // Training/Testing data
  vector<Tensor*> inputs;
  vector<Tensor*> targets;
  bool doTest;
  vector<Tensor*> testInputs;
  vector<Tensor*> testTargets;

  // Neuron data
  Tensor *aout, *zout, *deltas;
  Neuron** layers;

  // Helper functions
  inline void createArrays(vector<int>& neurons);
  inline void feedForward();
  inline bool checkMax(const Tensor& target);
  inline double error(const Tensor& target);
  inline void outputError(const Tensor& target);
  inline void backPropagate();
  inline void gradientDescent();
  inline void clearMatrices();
  inline bool checkStart(int& NData);
};

#endif
