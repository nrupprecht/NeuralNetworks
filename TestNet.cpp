#include "TestNet.h"

Network::Network() : layers(0), aout(0), zout(0), deltas(0) {};

Network::~Network() {
  if (layers) delete [] layers;
  if (aout) delete [] aout;
  if (zout) delete [] zout;
  if (deltas) delete [] deltas;
}

void Network::run(vector<Matrix*> inputs, vector<Matrix*> targets) {
  if (inputs.size() != targets.size()) throw 1;

  int NData = 50; // inputs.size()

  vector<int> l1s; l1s.push_back(784); l1s.push_back(1);
  vector<int> l2s; l2s.push_back(30); l2s.push_back(1);
  vector<int> l3s; l3s.push_back(10); l3s.push_back(1);

  int NLayers = 2;
  total = NLayers + 1;

  layers = new Neuron*[NLayers+1];
  layers[0] = 0;
  layers[1] = new Sigmoid(l1s, l2s);
  layers[2] = new Sigmoid(l2s, l3s);
  aout = new Matrix[NLayers+1];
  zout = new Matrix[NLayers+1];
  deltas = new Matrix[NLayers+1];
  
  // Set up aout and deltas
  aout[0].resize(784, 1); //576
  deltas[1].resize(30, 1);
  aout[1].resize(30,1);
  deltas[2].resize(10,1);
  aout[2].resize(10,1);

  // Constants
  int batchSize = 10;
  int nBatches = NData/batchSize;
  int leftOver = NData - nBatches*batchSize;
  int iterations = 10;
  double factor = 0.1/batchSize;

  // Run
  for (int iter=0; iter<iterations; iter++) {
    int trainCorrect = 0;
    clock_t start = clock();
    for (int batch=0; batch<nBatches; batch++) {
      for (int n=0; n<batchSize; n++) {
	int index = batch*batchSize + n;

	aout[0].qref(*inputs.at(index)); // Reference input
	// Feed forward
	for (int j=1; j<total; j++) layers[j]->feedForward(aout[j-1], aout[j]);

	// Check if correct
	if (checkMax(*targets.at(index))) trainCorrect++;
	// Get output error
	subtract(aout[total-1], *targets.at(index), deltas[total-1]);
	// Back prop
	for (int j=NLayers; j>1; j--) 
	  layers[j]->backPropagate(deltas[j], deltas[j-1], aout[j-1]);
	// Update weight and bias deltas
	for (int j=1; j<=NLayers; j++) 
	  layers[j]->updateDeltas(aout[j-1], deltas[j]);

	aout[0].qrel();
      }
      // Gradient descent
      for (int i=1; i<total; i++) layers[i]->gradientDescent(factor);
      // Clear layers and matrices
      for (int i=1; i<total; i++) {
	layers[i]->clear();
	deltas[i].zero();
      }

    }
    clock_t end = clock();
    cout << "Iteration " << iter << ": " << static_cast<double>(end-start)/CLOCKS_PER_SEC << " seconds." << endl;
      cout << "Training Set: " << trainCorrect << "/" << NData << " (" << 100*(double)trainCorrect/NData << "%)" << endl;   
  }
}

inline bool Network::checkMax(const Matrix& target) {
  // Assumes output, target are vectors
  if (aout[total-1].getRows()!=target.getRows())
    throw 1; // Make a real error for this sometime
  int t_index = -2; double t_max = -1e6;
  int o_index = -1; double o_max = -1e6;
  for (int i=0; i<target.getRows(); i++) {
    if (target.at(i,0)>t_max) {
      t_max = target.at(i,0);
      t_index = i;
    }
    if (aout[total-1].at(i,0)>o_max) {
      o_max = aout[total-1].at(i,0);
      o_index =i;
    }
  }
  return t_index==o_index;
}
