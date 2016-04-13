/// Network.cpp - Implements Network functions
/// Nathaniel Rupprecht 2016
///

#include "Network.h"

// Squaring function
inline double sqr(double x) { return x*x; }

Network::Network() : initialized(false), aout(0), zout(0), deltas(0), total(0), fnct(0), dfnct(0), rate(0.01), factor(0.), L2const(0.), L2factor(0.), trainingIters(100), minibatch(10), display(true), doTest(true) {};

Network::~Network() {
  if (aout) delete [] aout;
  if (zout) delete [] zout;
  if (deltas) delete [] deltas;
}

void Network::createFeedForward(vector<int> neurons, function F, function DF) {
  this->neurons = neurons;
  fnct = F; dfnct = DF;
  total = static_cast<int>(neurons.size());
  layers = new Neuron*[total];
  // Initialize layers
  layers[0] = 0;
  for (int i=1; i<total; i++) {
    vector<int> in_v; 
    in_v.push_back(neurons.at(i-1)); in_v.push_back(1); 
    vector<int> out_v;
    out_v.push_back(neurons.at(i)); out_v.push_back(1);
    layers[i] = new Sigmoid(in_v, out_v);
  }

  // Initialize vectors/matrices
  aout = new Matrix[total];
  zout = new Matrix[total];
  deltas = new Matrix[total];

  // Set vector/matrix sizes
  aout[0].resize(neurons.at(0), 1);
  zout[0].resize(neurons.at(0), 1);
  for (int i=1; i<total; i++) {
    aout[i].resize(neurons.at(i), 1);
    zout[i].resize(neurons.at(i), 1);
    deltas[i].resize(neurons.at(i), 1);
  }
  // The network has been initialized
  initialized = true;
}

void Network::train(int NData) {
  if (!checkStart(NData)) return;

  int nBatches = NData/minibatch;
  int leftOver = NData % minibatch;

  clearMatrices(); // Initial clear
  for (int iter=0; iter<trainingIters; iter++) {
    double aveError = 0;
    int trainCorrect = 0;
    // Start Timing
    clock_t start = clock();
    for (int i=0; i<nBatches; i++) {
      factor = rate/minibatch;
      L2factor = L2const * rate;
      for (int j=0; j<minibatch; j++) {
	int index = i*minibatch+j;
	aout[0].qref(*inputs.at(index)); // Reference input
	feedForward();
	// Check if was correct
	if (checkMax(*targets.at(index)))
	  trainCorrect++;
	// Backpropagate
	aveError += error(*targets.at(index));
	outputError(*targets.at(index));
	backPropagate();
	aout[0].qrel(); // Release reference
      }
      // Gradient descent
      gradientDescent();
      clearMatrices();
    }
    // Catch anything left out of a minibatch, make it its own minibatch
    factor = rate/leftOver;
    if (leftOver>0) {
      for (int i=0; i<leftOver; i++) {
	int index = NData-leftOver;
	aout[0].qref(*inputs.at(index)); // Reference input
	feedForward();
	// Check if was correct
        if (checkMax(*inputs.at(i)))
          trainCorrect++;
	// Backpropagate
	aveError += error(*targets.at(index));
	outputError(*targets.at(index));
	backPropagate();
	aout[0].qrel(); // Release reference
      }
      gradientDescent();
      clearMatrices();
    }

    // Iteration finished
    clock_t end = clock();
    if (display) { // Display iteration summary
      cout << "Iteration " << iter+1 << ": " << static_cast<float>(end - start)/CLOCKS_PER_SEC << " seconds." << endl;
      cout << "Ave Error: " << aveError/inputs.size() << endl;
      cout << "Training Set: " << trainCorrect << "/" << inputs.size() << " (" << 100.*static_cast<double>(trainCorrect)/inputs.size() << "%)" << endl;
      // See how we do on the test set
      if (doTest && !testInputs.empty()) {
	int correct = 0;
	for (int i=0; i<testInputs.size(); i++) {
	  aout[0].qref(*testInputs.at(i));
	  feedForward();
	  if (checkMax(*testTargets.at(i)))
	    correct++;
	  aout[0].qrel();
	}
	cout << "Test Set: " << correct << "/" << testInputs.size() << " (" << 100.*static_cast<double>(correct)/testInputs.size() << "%)\n";
      }
      // More later
      // cout << weights[1].norm() << endl;
      // Ending seperator
      cout << endl;
    }

    if (false) { 
      // Checks how many of the training examples the network gets right
      int correct = 0;
      for (int i=0; i<NData; i++) {
	aout[0].qref(*inputs.at(i));
	feedForward();
	if (checkMax(*targets.at(i)))
	  correct++;
	aout[0].qrel();
      }
      cout << "Training Set: " << correct/static_cast<double>(NData);
    }
  }
  cout << "Training over." << endl;
}

Matrix Network::feedForward(Matrix& input) {
  aout[0].qref(input);
  for (int i=1; i<total; i++)
    layers[i]->feedForward(aout[i-1], aout[i], zout[i]);
  aout[0].qrel();
  return aout[total-1];
}

inline void Network::feedForward() {
  for (int i=1; i<total; i++)
    layers[i]->feedForward(aout[i-1], aout[i], zout[i]);
}

/// Checks if the maximum entry of target corresponds to the
/// maximum entry of the result ( aout[total-1] )
inline bool Network::checkMax(const Matrix& target) {
  // Assumes output, target are vectors
  if (aout[total-1].getRows()!=target.getRows())
    throw 1; // Make a real error for this sometime
  int t_index = 0; double t_max = -1e6;
  int o_index = 0; double o_max = -1e6;
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

/// Squared error
inline double Network::error(const Matrix& target) {
  double error = 0;
  for (int i=0; i<target.getRows(); i++)
    error += sqr(target.at(i,0) - aout[total-1].at(i,0));
  return error;
}

/// This error is the cross entropy
inline void Network::outputError(const Matrix& target) {
  subtract(aout[total-1], target, deltas[total-1]);
}

inline void Network::backPropagate() {
  for (int j=total-1; j>1; j--)
    layers[j]->backPropagate(deltas[j], deltas[j-1], zout[j-1]);
  // Update weight and bias deltas
  for (int j=1; j<total; j++)
    layers[j]->updateDeltas(aout[j-1], deltas[j]);
}

inline void Network::gradientDescent() {
  for (int i=1; i<total; i++) layers[i]->gradientDescent(factor);
}

inline void Network::clearMatrices() {
  for (int i=1; i<total; i++) {
    layers[i]->clear();
    deltas[i].zero();
  }
}

inline bool Network::checkStart(int& NData) {
  if (inputs.size()!=targets.size()) {
    cout << "Training Input size (" << inputs.size() << "and Target size (" << targets.size() << ") do not match." << endl;
    return false; // Mismatch
  }
  if (testInputs.size()!=testTargets.size() && doTest) {
    cout << "Test Input size (" << testInputs.size() << ") and Target size (" << testTargets.size() << ") do not match." << endl;
    return false; // Mismatch
  }
  if (!initialized) {
    cout << "Network uninitialized" << endl;
    return false; // Uninitialized
  }
  if (NData<0 || NData>inputs.size()) NData = inputs.size();

  // Announcement
  cout << "Training data size: " << NData << endl;
  int complexity = 0, biases = 0;
  for (int i=1; i<neurons.size(); i++) complexity += neurons.at(i)*neurons.at(i-1);
  for (int i=1; i<neurons.size(); i++) biases += neurons.at(i);
  cout << "Net complexity: Weights: " << complexity << ", Biases: " << biases << endl << endl;

  return true;
}
