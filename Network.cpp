/// Network.cpp - Implements Network functions
/// Nathaniel Rupprecht 2016
///

#include "Network.h"

// Squaring function
inline double sqr(double x) { return x*x; }

Network::Network() : initialized(false), aout(0), zout(0), deltas(0), trainMarker(0), total(0), fnct(0), dfnct(0), rate(0.01), factor(0.), L2const(0.), L2factor(0.), trainingIters(100), minibatch(10), display(true), doTest(true), checkCorrect(true), calcError(true), testCorrect(0), rank(0), size(1) {
  //MPI_Init(&argc, &argv);
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
};

Network::~Network() {
  deleteArrays();
}

inline void Network::deleteArrays() {
  if (aout) delete [] aout;
  if (zout) delete [] zout;
  if (deltas) delete [] deltas;
  if (trainMarker) delete [] trainMarker;
}

inline void Network::createArrays(vector<int>& neurons) {
  this->neurons = neurons;
  total = static_cast<int>(neurons.size());
  layers = new Neuron*[total];
  
  // Create Tensor arrays
  aout = new Tensor[total];
  zout = new Tensor[total];
  deltas = new Tensor[total];
  trainMarker = new bool[total];

  // Set vector/matrix sizes
  aout[0].resize(neurons.at(0), 1);
  zout[0].resize(neurons.at(0), 1);
  for (int i=1; i<total; i++) {
    aout[i].resize(neurons.at(i), 1);
    zout[i].resize(neurons.at(i), 1);
    deltas[i].resize(neurons.at(i), 1);
  }
  // Set trainMarker array
  for (int i=0; i<total; i++) trainMarker[i] = true;
}

void Network::printDescription() {
  cout << layers[1]->getInShape() << " --> ";
  for (int i=1; i<total; i++) {
    cout << layers[i]->getOutShape();
    if (i!=total-1) cout << " --> ";
  }
  cout << endl;
  cout << "Using " << size << " processes." << endl;
}

void Network::createFeedForward(vector<int>& neurons, function F, function DF) {
  deleteArrays();
  createArrays(neurons);
  // Set up layers
  layers[0] = 0;
  for (int i=1; i<total; i++) {
    Shape in_v(neurons.at(i-1), 1);
    Shape out_v(neurons.at(i), 1);
    layers[i] = new Sigmoid(in_v, out_v);
  }

  // The network has been initialized
  initialized = true;
}

void Network::createCommonTensorPool() {
  for (int i=1; i<total; i++)
    for (auto T : layers[i]->getCommon())
      commonTensors.push_back(T);
}

void Network::createAutoEncoder(vector<int>& neur, function F, function DF) {
  vector<int> N;
  for (int i=0; i<neur.size(); i++) N.push_back(neur.at(i));
  for (int i=neur.size()-2; i>=0; i--) N.push_back(neur.at(i));
  createArrays(N);

  // Initialize layers
  layers[0] = 0;
  int i;
  for (i=1; i<=neur.size(); i++) {
    Shape in_v(N.at(i-1), 1);
    Shape out_v(N.at(i), 1);
    layers[i] = new Sigmoid(in_v, out_v);
  }
  int mid = i;
  for (int c=1; i<total; i++, c++) {
    Shape in_v(N.at(i-1), 1);
    Shape out_v(N.at(i), 1);
    
    Sigmoid* S = new Sigmoid(in_v, out_v);
    S->setTensor(0, layers[mid-c]->getTensor(0));
    S->setTransposed(true);

    layers[i] = S;
  }
  
  initialized = true;
  checkCorrect = false;
}

void Network::train(int NData) {
  if (!checkStart(NData)) return;

  if (minibatch<=0 || minibatch>NData) minibatch = NData;
  int nBatches = NData/minibatch;
  int leftOver = NData % minibatch;
  int outSize = targets.at(0)->size();
  double invErrNorm = 1.0/(NData*outSize);
  clearMatrices(); // Initial clear
  for (int iter=0; iter<trainingIters; iter++) {
    double aveError = 0;
    trainCorrect = 0;
    // Start Timing
    clock_t start = clock();
    factor = rate/minibatch;
    L2factor = L2const * rate;
    for (int i=0; i<nBatches; i++) {
      trainMinibatch(i*minibatch, minibatch, aveError);
      gradientDescent();
      clearMatrices();
    }
    // Catch anything left out of a minibatch, make it its own minibatch
    factor = leftOver==0 ? 0 : rate/leftOver;
    if (leftOver>0) {
      trainMinibatch(NData-leftOver, leftOver, aveError);
      gradientDescent();
      clearMatrices();
    }
    // Iteration finished
    clock_t end = clock();
    if (invErrNorm!=0) aveError*=invErrNorm;
    // Check on test set
    if (doTest && testInputs.size()>0) {
      checkTestSet();
      testPercentRec.push_back((double)testCorrect/testInputs.size());
    }
    // Display iteration summary
    timeRec.push_back((double)(end-start)/CLOCKS_PER_SEC);
    if (display) printData(iter+1, (float)(end-start)/CLOCKS_PER_SEC, aveError);
    // Record data
    if (calcError) errorRec.push_back(aveError);
    if (checkCorrect) trainPercentRec.push_back((double)trainCorrect/inputs.size());
  }
  if (display) cout << "Training over." << endl;
}

void Network::trainMPI(int NData) {
  if (size==1) {
    train(NData);
    return;
  }
  
  if (rank==0 && !checkStart(NData)) return;
  if (rank!=0 && !checkStart(NData,true)) return;
  
  // Create common tensor structure
  createCommonTensorPool();
  
  if (minibatch<=0 || minibatch>NData) minibatch = NData;
  int nBatches = NData/minibatch;
  int leftOver = NData % minibatch;
  
  int outSize = targets.at(0)->size();
  double invErrNorm = 1.0/(NData*outSize);

  // Find out how much of a minibatch to do
  int num = ceil((double)minibatch/size);
  int shift = rank*num;
  num = min(num, minibatch-shift);
  num = max(num, 0);

  clock_t start, end;
  clearMatrices(); // Initial clear
  for (int iter=0; iter<trainingIters; iter++) {
    double aveError = 0;
    trainCorrect = 0;
    // Start Timing
    if (rank==0) start = clock();
    factor = rate/minibatch;
    L2factor = L2const * rate;
    for (int i=0; i<nBatches; i++) {
      trainMinibatch(i*minibatch+shift, num, aveError);
      MPI_Barrier( MPI_COMM_WORLD );
      // Gather and add delta matrices
      if (size>1)
	for (auto T : commonTensors)
	  MPI_Allreduce(MPI_IN_PLACE, T->getArray(), T->size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      // Do a gradient descent
      gradientDescent();
      clearMatrices();
    }
    // Catch anything left out of a minibatch, make it its own minibatch
    /*
    factor = rate/leftOver;
    if (leftOver>0) {
      trainMinibatch(NData-leftOver, leftOver, aveError);
      
      MPI_Barrier( MPI_COMM_WORLD );
      // Gather and add delta matrices
      for (auto T : commonTensors)
        MPI_Allreduce(MPI_IN_PLACE, T->getArray(), T->size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);      
      gradientDescent();
      clearMatrices();
    }
    */
    
    // Get stats from all the processes
    MPI_Barrier( MPI_COMM_WORLD );
    if (size>1) {
      MPI_Allreduce(MPI_IN_PLACE, &trainCorrect, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &aveError, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    // Iteration finished
    if (rank==0) {
      end = clock();
      timeRec.push_back((double)(end-start)/CLOCKS_PER_SEC);
      aveError*=invErrNorm;
      // Check on test set
      if (doTest && testInputs.size()>0) {
        checkTestSet();
        testPercentRec.push_back((double)testCorrect/testInputs.size());
      }
      // Display iteration summary
      if (display) printData(iter+1, (float)(end-start)/CLOCKS_PER_SEC, aveError);
      // Record data
      if (calcError) errorRec.push_back(aveError);
      if (checkCorrect) trainPercentRec.push_back((double)trainCorrect/inputs.size());
    }
    MPI_Barrier( MPI_COMM_WORLD ); // Wait to start the next iteration
  }

  // Finalize MPI
  MPI_Barrier( MPI_COMM_WORLD );
  //MPI_Finalize();

  if (rank==0 && display) cout << "Training over." << endl;
}

Tensor Network::feedForward(Tensor& input) {
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

double Network::getAveTime() {
  if (rank==0) {
    if (timeRec.size()==0) return -1;
    double ave;  
    for (auto t : timeRec) ave += t;
    return ave/timeRec.size();
  }
}

/// Checks if the maximum entry of target corresponds to the
/// maximum entry of the result ( aout[total-1] )
inline bool Network::checkMax(const Tensor& target) {
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
inline double Network::sqrError(const Tensor& target) {
  double error = 0;
  for (int i=0; i<target.getRows(); i++)
    error += sqr(target.at(i,0) - aout[total-1].at(i,0));
  return error;
}

/// This error is the cross entropy
inline void Network::outputError(const Tensor& target) {
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
  for (int i=1; i<total; i++) 
    if (trainMarker[i]) 
      layers[i]->gradientDescent(factor);
}

inline void Network::clearMatrices() {
  for (int i=1; i<total; i++) {
    layers[i]->clear();
    deltas[i].zero();
  }
}

inline bool Network::checkStart(int& NData, bool quiet) {
  if (inputs.size()!=targets.size()) {
    if (!quiet) cout << "Training Input size (" << inputs.size() << "and Target size (" << targets.size() << ") do not match." << endl;
    return false; // Mismatch
  }
  if (testInputs.size()!=testTargets.size() && doTest) {
    if (!quiet) cout << "Test Input size (" << testInputs.size() << ") and Target size (" << testTargets.size() << ") do not match." << endl;
    return false; // Mismatch
  }
  if (!initialized) {
    if (!quiet) cout << "Network uninitialized" << endl;
    return false; // Uninitialized
  }
  if (NData<0 || NData>inputs.size()) NData = inputs.size();
  if (NData==0) {
    if (!quiet) cout << "No data to train on" << endl;
    return false; // No data
  }

  // Announcement
  if (display && !quiet) {
    cout << "Training data size: " << NData << endl;
    int complexity = 0, biases = 0;
    for (int i=1; i<neurons.size(); i++) complexity += neurons.at(i)*neurons.at(i-1);
    for (int i=1; i<neurons.size(); i++) biases += neurons.at(i);
    cout << "Net complexity: Weights: " << complexity << ", Biases: " << biases << endl << endl;
  }
  return true;
}

inline void Network::trainMinibatch(int base, int num, double& aveError) {
  for (int j=0; j<num; j++) {
    int index = base+j;
    aout[0].qref(*inputs.at(index)); // Reference input
    feedForward();
    // Check if was correct
    if (checkCorrect && checkMax(*targets.at(index))) trainCorrect++;
    // Calculate the error
    if (calcError) aveError += sqrError(*targets.at(index));
    // Backpropagate
    outputError(*targets.at(index));
    backPropagate();
    aout[0].qrel(); // Release reference
  }
}

inline void Network::printData(int iter, float time, double aveError) {
  cout << "Iteration " << iter << ": " << time << " seconds." << endl;
  if (calcError) cout << "Ave Error: " << aveError << endl;
  if (checkCorrect)
    cout << "Training Set: " << trainCorrect << "/" << inputs.size() << " (" << 100.*\
      static_cast<double>(trainCorrect)/inputs.size() << "%)" << endl;
  // See how we do on the test set
  if (doTest && !testInputs.empty()) {
    /*
    int correct = 0;
    for (int i=0; i<testInputs.size(); i++) {
      aout[0].qref(*testInputs.at(i));
      feedForward();
      if (checkMax(*testTargets.at(i))) correct++;
      aout[0].qrel();
    }
    */
    cout << "Test Set: " << testCorrect << "/" << testInputs.size() << " (" << 100.*static_cast<double>(testCorrect)/testInputs.size() << "%)\n";
  }
  cout << endl;
}

inline void Network::checkTestSet() {
  testCorrect = 0;
  for (int i=0; i<testInputs.size(); i++) {
    aout[0].qref(*testInputs.at(i));
    feedForward();
    if (checkMax(*testTargets.at(i))) testCorrect++;
    aout[0].qrel();
  }
}
