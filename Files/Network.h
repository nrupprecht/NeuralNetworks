//
//  Network.h
//  Network
//
//  Created by Nathaniel Rupprecht on 6/27/15.
//  Copyright (c) 2015 Nathaniel Rupprecht. All rights reserved.
//

#ifndef __Network__Network__
#define __Network__Network__

#include <math.h>
#include <vector>
using std::vector;

#include <fstream>
using std::ofstream;
using std::ifstream;
using std::string;

#include <sstream>
using std::stringstream;

#include <thread> // Multithreading
#include <unistd.h> // For usleep

#include "Empack.h"
#include "Stopwatch.h" // For timing

// The Four Equations of Backpropagation:
// (Notes taken from http://neuralnetworksanddeeplearning.com/chap2.html)
// For a network with L layers
// Definitions:
//      delta^{i}_{j}   -- Error of the jth node in the ith layer
//      C               -- Cost function
//      z^{i}_{j}       -- The weighted output of the jth node in the ith layer
//      a^{i}_{j}       -- The activation (output) of the jth node in the ith layer
//      w^{i}           -- The weight matrix of the ith layer
//      A^B             -- Hadamard product of A and B
//      dsigmoid        -- Derivative of the sigmoid function
//      (A)T            -- Transpose of A
//
// 1) Error in the output layer: delta^{L}_{j} = ∂C/∂(a^{L}_{j}) dsigmoid(z^{L}_{j})
//    In matrix form: delta^{L} = ∂_{a} C ^ dsigmoid(z^{L})
// 2) Error in the hidden layers: delta^{i} = ( ( w^{i+1} )T * delta^{i+1} ) ^ dsigmoid(z^{i})
// 3) Rate of change of cost with respect to biases: ∂C/∂b = delta
// 4) Rate of change of cost with respect to weights: ∂C/∂w = a(in) * delta(out)

// A helper printing function
inline void print(float *array, uint len) {
    for(int i=0; i<len; i++)
        cout << array[i] << " ";
    cout << endl;
}

// A helper function to find what the max entry's value is
inline float max(float *array, uint len) {
    if(len == 0) return 0.f;
    float max = array[0];
    for(int i=0; i<len; i++) {
        if(max < array[i]) max = array[i];
    }
    return max;
}

// A helper function to find which entry is the max
inline uint maxIndex(float *array, uint len) {
    if(len == 0) return 0.f;
    float max = array[0];
    uint index = 0;
    for(int i=0; i<len; i++) {
        if(max < array[i]) {
            max = array[i];
            index = i;
        }
    }
    return index;
}

inline std::string toString(float num) {
    stringstream stream;
    stream << num;
    string S;
    stream >> S;
    return S;
}

template <typename T>
void safeDelete(T* pt) {
    if(pt != 0) {
        delete [] pt;
        pt = 0;
    }
}

class Network {
public:
    Network();
    Network(uint layers, uint* sizes);
    ~Network();
    
    // Training function
    void trainOnDataset(bool (*checkFunc) (const Matrix<>&, const Matrix<>&),
                        vector<float*> inputVectors, vector<float*> outputVectors,
                        vector<float*> testData = vector<float*>(), vector<float*> testLabel  = vector<float*>());
    void mtTrainOnDataset(bool (*checkFunc) (const Matrix<>&, const Matrix<>&),
                          vector<float*> inputVectors, vector<float*> outputVectors,
                          vector<float*> testData = vector<float*>(), vector<float*> testLabel  = vector<float*>());
    
    void autoEncoder(vector<float*> inputVectors);
    
    void saveNetwork(string fileName);
    void loadNetwork(string fileName);
    void loadNetworkAndAdd(string fileName, uint newsize);
    
    Matrix<> evaluate(const Matrix<>& input); // Evaluate an input
    
    inline static void sigmoid(float& z) { z = 1.f/(1+exp(-z));}
    
    // Accessors
    Matrix<> getError(uint layer);
    Matrix<>& getWeights(uint layer);
    Matrix<>& getBiases(uint layer);
    Matrix<> getOutput();
    string getCostRecord();
    uint getComplexity();
    uint getLayers() {return layers;}
    const uint* getSizes() {return sizes;}
    
    // Mutators
    void setMiniBatchSize(uint size);
    void setEta(float eta);
    void setEtaDFactor(float Edf)   {etaDFactor = Edf;}
    void setEtaDecay(bool decay)    {etaDecay = decay;}
    void setLambda(float lambda);
    void setTrainingIterations(uint iters) {trainingIterations = iters;}
    void setNThreads(uint nThreads) {this->nThreads = nThreads;}
    void setStoreBest(bool store) {storeBest = store;}
    void setWorkFromBest(bool work) {workFromBest = storeBest = true;}
    void setPeriodicSave(bool set, uint spacing=1) {periodicSave = set; saveSpacing=spacing;}
    void popLayer();
    void addLayers(uint layers, uint* sizes);
    
    // Exception classes
    class LayerNonexistent {};
    class TrainingDataMismatch {};
    
private:
    // Options
    bool etaDecay;      // Whether we want eta to decay
    bool storeBest;     // Whether we want to store the bset weights and biases
    bool workFromBest;  // If true, if when we update the network, we get worse performace, we go back and try again
    bool periodicSave;
    uint saveSpacing;
    
    // The (i)th entry represents the weights between the (i-1)th layer and the (i)th layer
    Matrix<> *weights;    // Matrices representing the weights between the layers
    Matrix<> *biases;     // Bias vectors, the biases[0] entry has no meaning since inputs have no biases
    Matrix<> *zoutputs;   // Weighted output (pre sigmoid)
    Matrix<> *aoutputs;   // An array (of length [layers]) of column vectors representing the activation \
                        output at each layer (outputs[0] represents the input)
    Matrix<> *delta;      // Errors of each node
    // Gradient descent
    Matrix<> *weightDelta;    // Change in weights to do at the gradient descent state
    Matrix<> *biasDelta;      // Change in biases to do at the gradient descent stage
    
    // Training functions
    float getCost(Matrix<>& expected);
    void feedForward(const Matrix<>& input);
    void computeOutputError(const Matrix<>& output);
    void backPropagate();
    void gradientDescent();
    
    // Utility functions
    inline static void dsigmoid(float& z) { // Set z to d/dz(sigmoid) | z
        float sig = z;
        sigmoid(sig);
        z = sig*(1-sig); // We do this to avoid getting any nan's
    }
    
    inline void setUp();
    inline void setHelperMatrices();
    inline void clearHelperMatrices();
    inline void loadData(string fileName);
    
    // Multithreading
    void threadTraining();
    uint nThreads;                  // Number of threads to use
    std::thread *pool;              // Thread pool
    std::mutex taskLock;            // Lock for doing backpropagatoin
    std::mutex idMutex;             // Mutex for threads getting ID's
    uint ident;                     // What id the next thread should take
    bool *status;                   // For communicating with and controlling the threads
    bool trainingDone;              // True when we are done training and the threads can be retired
    std::pair<uint, uint> *ranges;  // For telling threads which sections of data to loop over
    vector<uint> indexList;         // We shuffle this before each epoch
    bool (*checkFunc_t) (const Matrix<>&, const Matrix<>&); // The check function
    uint *threadCorrect;            // The number of samples each thread got correct
    // MT Helper functions
    inline void feedForward(const Matrix<>& input, Matrix<>* aoutputs_t, Matrix<>* zoutputs_t);
    inline float getCost(Matrix<> *expected, Matrix<>* aoutputs_t);
    inline void computeOutputError(const Matrix<>&, const Matrix<>*, Matrix<>*, Matrix<>*);
    inline void backPropagate(Matrix<>*, Matrix<>*, Matrix<>*, Matrix<>*, Matrix<>*);
    inline void gradientDescent(Matrix<>* weightDelta_t, Matrix<>* biasDelta_t);
    
    // Training variables
    uint miniBatchSize;
    uint trainingIterations;
    uint NData;         // The amount of training data we have
    float eta;          // Learning rate
    float factor;       // eta/miniBatchSize
    float lambda;       // L2 regularization factor
    float L2factor;     // eta * L2 regularization / miniBatchSize
    float etaDFactor;   // The factor that eta decays by each iteration
    vector<float> CostRecord;
    
    // Store input/labels, test/labels
    Matrix<> *inputSet;
    Matrix<> *outputSet;
    Matrix<> *testInput;
    Matrix<> *testOutput;
    
    // Characteristics
    uint *sizes;        // sizes[0] = number of inputs, sizes[layers-1] = number of outputs
    bool *modify;
    uint layers;
};

#endif /* defined(__Network__Network__) */
