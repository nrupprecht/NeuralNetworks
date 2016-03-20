//
//  Network.cpp
//  Network
//
//  Created by Nathaniel Rupprecht on 6/27/15.
//  Copyright (c) 2015 Nathaniel Rupprecht. All rights reserved.
//

#include "Network.h"

#include <algorithm> // For std::shuffle

Network::Network() {
    sizes = 0;
    layers = 0;
    
    setUp();
    
    weights = 0;
    biases = 0;
    
    zoutputs = 0;
    aoutputs = 0;
    delta = 0;
    weightDelta = 0;
    biasDelta = 0;
}

Network::Network(uint layers, uint* sizes) {
    // layers - the number of layers (and the size of the array sizes)
    // size - the sizes of each layer
    this->sizes = sizes;
    this->layers = layers;
    
    setUp();
    
    // Create weights and biases
    weights = new Matrix<>[layers];       // weights[0] is null
    biases = new Matrix<>[layers];        // biases[0] is null
    
    for (int i=1; i<layers; i++) {
        // Initialize random weights
        weights[i] = Matrix<>::randMatrix(sizes[i], sizes[i-1]);
        weights[i] *= 1.f/sqrt(sizes[i-1]);
        biases[i] = Matrix<>::randMatrix(sizes[i], 1);
    }
    
    setHelperMatrices();
}

Network::~Network() {
    if(pool != 0)
        for(int i=0; i<nThreads; i++)
            if(pool[i].joinable())
                pool[i].join();
    // Delete dynamic memory
    safeDelete(weights);
    safeDelete(weightDelta);
    safeDelete(biases);
    safeDelete(biasDelta);
    safeDelete(pool);
}

inline void Network::setUp() {
    miniBatchSize = 10;
    trainingIterations = 30;
    NData = 1;
    eta = 0.5f;
    lambda = 0.f;
    factor = eta/miniBatchSize;
    L2factor = eta*lambda;
    etaDFactor = 0.8f;
    etaDecay = false;
    
    nThreads = 4;
    pool = 0;
    status = 0;
    threadCorrect = 0;
    trainingDone = false;
    ident = 0;
    
    periodicSave = false;
    saveSpacing=1;
}

inline void Network::setHelperMatrices() {
    zoutputs = new Matrix<>[layers];      // zoutputs[0] represents the input
    aoutputs = new Matrix<>[layers];      // aoutputs[0] represents the sigmoid-ed input
    delta = new Matrix<>[layers];         // delta[0] is null
    weightDelta = new Matrix<>[layers];   // weightDelta[0] is null
    biasDelta = new Matrix<>[layers];     // biasDelta[0] is null
    
    for (int i=1; i<layers; i++) {
        weightDelta[i] = Matrix<>(sizes[i], sizes[i-1]);
        aoutputs[i] = Matrix<>(sizes[i],1); // Initialize a column vector
        zoutputs[i] = Matrix<>(sizes[i],1);
        biasDelta[i] = Matrix<>(sizes[i],1);
        delta[i] = Matrix<>(sizes[i],1);
    }
}

inline void Network::clearHelperMatrices() {
    // We don't need to clear delta[i]
    for(int i=0; i<layers; i++) {
        weightDelta[i].clear();
        biasDelta[i].clear();
    }
}

void Network::saveNetwork(string fileName) {
    ofstream fout(fileName);
    if(fout.fail()) {
        cout << "File failed to open";
        return;
    }
    else cout << "Saving file ... ";
    // Output magic number, number of layers
    fout << "1992 " << layers << " ";
    // Ouput sizes
    for(int i=0; i<layers; i++) fout << sizes[i] << " ";
    fout << endl;
    // Output Weights
    for(int i=1; i<layers; i++) {
        saveFormat(weights[i], fout);
        fout << endl;
    }
    // Output Biases
    for(int i=1; i<layers; i++) {
        saveFormat(biases[i], fout);
        fout << endl;
    }
    fout << "10101"; // Footer
    fout.close(); // Close the filestream
    cout << "Done." << endl;
}

inline void Network::loadData(string fileName) {
    ifstream fin(fileName);
    
    if(fin.fail()) {
        cout << "File failed to open" << endl;
        return;
    }
    uint magic, L, footer;
    // Get magic number, layers
    fin >> magic >> L;
    if(magic != 1992 || L == 0) return; // Not a good file
    if(L != layers) { // If we need to resize our variables
        layers = L;
        sizes = new uint[layers];
        weights = new Matrix<>[layers];
        biases = new Matrix<>[layers];
    }
    // Load sizes
    for(int i=0; i<L; i++) fin >> sizes[i];
    // Load Weights
    for(int i=1; i<L; i++) weights[i] = loadFormat<float>(fin);
    // Load Biases
    for(int i=1; i<L; i++) biases[i] = loadFormat<float>(fin);
    fin >> footer;
    if(footer != 10101) // Critical error
        cout << "Error" << endl;
    fin.close(); // Close the filestream
}

void Network::loadNetwork(string fileName) {
    loadData(fileName); // Reset network variables
    setHelperMatrices();
}

void Network::loadNetworkAndAdd(string fileName, uint newsize) {
    loadData(fileName);
    //
    // Add a new layer between the last hidden layer and the output layer
    uint *newSizes = new uint[layers+1];
    for(int i=0; i<layers-1; i++)
        newSizes[i] = sizes[i];
    newSizes[layers-1] = newsize;
    newSizes[layers] = sizes[layers-1];
    // Swap arrays
    Matrix<>* newWeights = new Matrix<>[layers+1];
    Matrix<>* newBiases = new Matrix<>[layers+1];
    for(int i=1; i < layers-1; i++) {
        newWeights[i] = weights[i];
        newBiases[i] = biases[i];
    }
    // Initialize the new weights and biases
    newWeights[layers-1] = Matrix<>::randMatrix(newSizes[layers-1], newSizes[layers-2]);
    newWeights[layers-1] *= 1.f/sqrt(sizes[layers-2]);
    newWeights[layers] = Matrix<>::randMatrix(newSizes[layers], newSizes[layers-1]);
    newWeights[layers] *= 1.f/sqrt(sizes[layers-1]);
    newBiases[layers-1] = Matrix<>::randMatrix(newSizes[layers-1],1);
    newBiases[layers] = Matrix<>::randMatrix(newSizes[layers],1);
    
    layers = layers+1;
    //delete sizes;
    sizes = newSizes;
    //delete weights;
    weights = newWeights;
    //delete biases;
    biases = newBiases;
    
    setHelperMatrices();
}

Matrix<> Network::evaluate(const Matrix<>& input) {
    feedForward(input);
    return aoutputs[layers-1];
}

void Network::trainOnDataset(bool (*checkFunc) (const Matrix<>&, const Matrix<>&),
                             vector<float*> inputVectors, vector<float*> outputVectors,
                             vector<float*> testData, vector<float*> testLabel)
{
    if(inputVectors.size() != outputVectors.size()) throw TrainingDataMismatch();
    if(testData.size() != testLabel.size()) throw TrainingDataMismatch();
    if(inputVectors.size() == 0) return; // No data to train on
    
    // Prepair input and output data as matrices
    NData = static_cast<uint>(inputVectors.size());
    inputSet = new Matrix<>[NData];
    outputSet = new Matrix<>[NData];
    testInput = 0;
    testOutput = 0;
    
    // Process training data
    for(int i=0; i<inputVectors.size(); i++) {
        inputSet[i] = Matrix<>(inputVectors.at(i),sizes[0]);
        outputSet[i] = Matrix<>(outputVectors.at(i),sizes[layers-1]);
    }
    
    // Process testing data
    if(!testData.empty()) {
        testInput = new Matrix<>[testData.size()];
        testOutput = new Matrix<>[testData.size()];
        for(int i=0; i<testData.size(); i++) {
            testInput[i] = Matrix<>(testData.at(i), sizes[0]);
            testOutput[i] = Matrix<>(testLabel.at(i), sizes[layers-1]);
        }
    }
    
    CostRecord.clear();
    // Train stochastically
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,(int)inputVectors.size()-1);
    float minCost = MAXFLOAT, lastCorrect = 0;
    uint maxCorrect = 0;
    
    // A vector of indices that we can shuffle for our stochastic training
    vector<uint> indices;
    for(int i=0; i<inputVectors.size(); i++)
        indices.push_back(i);
    // Create a stopwatch for timing
    Stopwatch timer;
    // Train network
    uint NTrain = (NData/miniBatchSize) * miniBatchSize;
    for (int iter=0; iter<trainingIterations; iter++) {
        timer.start(); // Start timing
        // Shuffle our indices (stochastic training)
        std::shuffle(indices.begin(), indices.end(), generator);
        // Process a batch
        float cost = 0;
        uint correct = 0;
        for(int j=0; j<NData/miniBatchSize; j++) {
            clearHelperMatrices(); // We need to clear these every time since we only += to them.
            // Process a minibatch
            for(int i=0; i<miniBatchSize; i++) {
                uint index = indices.at(i+j*miniBatchSize); // Get index
                feedForward(inputSet[index]);
                // Check if correct
                if(checkFunc(aoutputs[layers-1], outputSet[index]))
                    correct++;
                // Compute cost
                cost += getCost(outputSet[index]); //getCost(outputSet[index]);
                computeOutputError(outputSet[index]);
                backPropagate();
            }
            gradientDescent(); // Update weights and biases with the deltas from the batch
        }
        
        timer.end();
        float aveCost = cost/NTrain;
        CostRecord.push_back(aveCost);
        cout << "Batch " << iter << ": Average cost: " << aveCost << endl;
        cout << "Batch took " << timer.time() << " seconds." << endl;
        if (aveCost < minCost) minCost = aveCost;
        cout << "Correct on training data: " << correct << '/' << inputVectors.size() <<
            " (" << 100.f*(float)correct/inputVectors.size() << "%)" << endl;
        
        // If we have testing data, use it
        if(!testData.empty()) {
            correct = 0;
            for(int i=0; i<testData.size(); i++) {
                feedForward(testInput[i]);
                if(checkFunc(aoutputs[layers-1], testOutput[i]))
                    correct++;
            }
            cout << "Correct on test set: " << correct << '/' << testData.size() << " ("
                << 100.f*(float)correct/testData.size() << "%)" << endl << endl;
            if(correct > maxCorrect) maxCorrect = correct;
            
            if(etaDecay && correct < lastCorrect ) {
                setEta(eta*etaDFactor);
                cout << "Changing eta to " << eta << endl;
            }
            lastCorrect = correct;
        }
        else if(etaDecay) {
            // Eta decay when there is no test data
        }
        cout << endl;
    }
    
    cout << "The minimum ave cost was " << minCost << endl;
    cout << "The most correct was " << maxCorrect << endl;
}

void Network::mtTrainOnDataset(bool (*checkFunc) (const Matrix<>&, const Matrix<>&),
                             vector<float*> inputVectors, vector<float*> outputVectors,
                             vector<float*> testData, vector<float*> testLabel)
{
    if(inputVectors.size() != outputVectors.size()) throw TrainingDataMismatch();
    if(testData.size() != testLabel.size()) throw TrainingDataMismatch();
    if(inputVectors.size() == 0) return; // No data to train on
    
    checkFunc_t = checkFunc;
    
    // Prepair input and output data as matrices
    NData = static_cast<uint>(inputVectors.size());
    inputSet = new Matrix<>[NData];
    outputSet = new Matrix<>[NData];
    testInput = 0;
    testOutput = 0;
    
    // Process training data
    for(int i=0; i<inputVectors.size(); i++) {
        inputSet[i] = Matrix<>(inputVectors.at(i),sizes[0]);
        outputSet[i] = Matrix<>(outputVectors.at(i),sizes[layers-1]);
    }
    
    // Process testing data
    if(!testData.empty()) {
        testInput = new Matrix<>[testData.size()];
        testOutput = new Matrix<>[testData.size()];
        for(int i=0; i<testData.size(); i++) {
            testInput[i] = Matrix<>(testData.at(i), sizes[0]);
            testOutput[i] = Matrix<>(testLabel.at(i), sizes[layers-1]);
        }
    }
    
    // Prepair index list
    indexList.clear();
    for(int i=0; i<inputVectors.size(); i++) indexList.push_back(i);
    
    // Create thread pool if neccessary
    if(pool == 0) {
        status = new bool[nThreads];
        pool = new std::thread[nThreads];
        ranges = new std::pair<uint,uint>[nThreads];
        threadCorrect = new uint[nThreads];
        
        for(int i=0; i<nThreads; i++) {
            status[i] = false;
            pool[i] = std::thread(&Network::threadTraining, this);
        }
    }
    
    // Set ranges for the threads
    uint delta = NData/nThreads;
    uint start=0, end = delta;
    for( int i=0; i<nThreads; i++ ) {
        if(i==nThreads-1) end = NData;
        ranges[i] = std::pair<uint,uint>(start,end);
        start += delta; end += delta;
    }
    
    Matrix<> *saveWeights = 0, *saveBiases = 0;
    if(storeBest) {
        saveWeights = new Matrix<>[layers];       // weights[0] is null
        saveBiases = new Matrix<>[layers];        // biases[0] is null
    }
    
    // Train data
    Stopwatch timer;
    uint maxCorrect = 0;
    uint lastCorrect = 0;
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,(int)inputVectors.size()-1);
    for (int iter=0; iter<trainingIterations; iter++) {
        timer.start(); // Start timing
        // Shuffle our indices (stochastic training)
        std::shuffle(indexList.begin(), indexList.end(), generator);
        
        for(int i=0; i<nThreads; i++) status[i] = true; // Start the threads
        
        // Wait for threads to finish
        bool done = false;
        while(!done) {
            usleep(100000); // Sleep 0.1 second as to not waste resources
            done = true;
            for(int i=0; i<nThreads; i++) if(status[i]) done = false; // Not all threads are done
        }
        
        timer.end();
        cout << "Batch " << iter << endl;
        cout << "Batch took " << timer.time() << " seconds." << endl;
        
        // Find the number the net got correct
        uint correct = 0;
        for(int id=0; id<nThreads; id++) correct += threadCorrect[id];
        
        cout << "Correct on training data: " << correct << '/' << inputVectors.size() <<
        " (" << 100.f*(float)correct/inputVectors.size() << "%)" << endl;
        
        // If we have testing data, use it
        if(!testData.empty()) {
            correct = 0;
            for(int i=0; i<testData.size(); i++) {
                feedForward(testInput[i]);
                if(checkFunc_t(aoutputs[layers-1], testOutput[i]))
                    correct++;
            }
            cout << "Correct on test set: " << correct << '/' << testData.size() << " ("
            << 100.f*(float)correct/testData.size() << "%)" << endl << endl;
            
            if(correct < lastCorrect && etaDecay) {
                eta *= etaDFactor;
                cout << "Reducing eta to " << eta << endl;
            }
            lastCorrect = correct;
        }
        
        // Keep records and (possibly) save and/or reset weights
        if(correct > maxCorrect) {
            maxCorrect = correct;
            if(storeBest) {
                for(int i=1; i<layers; i++) {
                    saveWeights[i] = weights[i];
                    saveBiases[i] = biases[i];
                }
            }
        }
        else if(workFromBest) { // Go back and try again
            cout << "Going back to best weight settings";
            eta *= etaDFactor; // Reduce eta
            for(int i=1; i<layers; i++) {
                weights[i] = saveWeights[i];
                biases[i] = saveBiases[i];
            }
        }
        
        if(periodicSave && iter%saveSpacing == 0) { // Time to save
            saveNetwork("backup.net");
        }
        cout << endl; // For formatting
    }
    
    cout << "Best: " << maxCorrect << endl;
    
    trainingDone = true;
}

void Network::autoEncoder(vector<float*> inputVectors) {
    if(inputVectors.size() == 0) return; // No data to train on
    
    // Prepair input and output data as matrices
    NData = static_cast<uint>(inputVectors.size());
    inputSet = new Matrix<>[NData];
    testInput = 0;
    
    // Process training data
    for(int i=0; i<inputVectors.size(); i++) {
        inputSet[i] = Matrix<>(inputVectors.at(i),sizes[0]);
    }
    
    CostRecord.clear();
    
    // Train stochastically
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,(int)inputVectors.size()-1);
    float minCost = MAXFLOAT;
    
    // A vector of indices that we can shuffle for our stochastic training
    vector<uint> indices;
    for(int i=0; i<inputVectors.size(); i++) indices.push_back(i);
    // Create a stopwatch for timing
    Stopwatch timer;
    // Train network
    uint NTrain = (NData/miniBatchSize) * miniBatchSize;
    for (int iter=0; iter<trainingIterations; iter++) {
        timer.start(); // Start timing
        // Shuffle our indices (stochastic training)
        std::shuffle(indices.begin(), indices.end(), generator);
        // Process a batch
        float cost = 0;
        for(int j=0; j<NData/miniBatchSize; j++) {
            clearHelperMatrices(); // We need to clear these every time since we only += to them.
            // Process a minibatch
            for(int i=0; i<miniBatchSize; i++) {
                uint index = indices.at(i+j*miniBatchSize); // Get index
                feedForward(inputSet[index]);                // Compute cost
                cost += getCost(inputSet[index]); //getCost(outputSet[index]);
                computeOutputError(inputSet[index]);
                backPropagate();
            }
            gradientDescent(); // Update weights and biases with the deltas from the batch
        }
        
        timer.end();
        float aveCost = cost/NTrain;
        CostRecord.push_back(aveCost);
        cout << "Batch " << iter << ": Average cost: " << aveCost << endl;
        cout << "Batch took " << timer.time() << " seconds." << endl << endl;
        if (aveCost < minCost) minCost = aveCost;
    }
    
    cout << "The minimum ave cost was " << minCost << endl;
}

void Network::threadTraining() {
    // Get an id
    uint ID;
    idMutex.lock();
    ID = ident;
    ident++;
    idMutex.unlock();
    
    // Each thread needs its own aoutput and zoutput matrices so as not to interfere with other threads
    // Each thread also needs its own delta matrices (or we need to use a mutex on the global delta arrays)
    Matrix<>* zoutputs_t = new Matrix<>[layers];      // // zoutputs[0] represents the input
    Matrix<>* aoutputs_t = new Matrix<>[layers];      // // aoutputs[0] represents the sigmoid-ed input
    Matrix<>* delta_t = new Matrix<>[layers];         // delta[0] is null
    Matrix<>* weightDelta_t = new Matrix<>[layers];   // weightDelta[0] is null
    Matrix<>* biasDelta_t = new Matrix<>[layers];     // biasDelta[0] is null
    for (int i=1; i<layers; i++) {
        weightDelta_t[i] = Matrix<>::randMatrix(sizes[i], sizes[i-1]);
        aoutputs_t[i] = Matrix<>(sizes[i],1); // Initialize a column vector
        zoutputs_t[i] = Matrix<>(sizes[i],1);
        biasDelta_t[i] = Matrix<>(sizes[i],1);
        delta_t[i] = Matrix<>(sizes[i],1);
    }
    
    // Do work whenever new work is given
    while(!trainingDone) {
        
        while(!status[ID]) {
            usleep(100000); // Sleep for 0.1s
            if(trainingDone) return;
        };  // Wait for signal
    
        // Get range
        auto range = ranges[ID];
        
        threadCorrect[ID] = 0; // Reset correctness count
        float cost = 0;
        uint start = range.first, end = range.second;
        for(int j=start; j<end-miniBatchSize; j+=miniBatchSize) {
            // Clear helper matrices
            for(int i=1; i<layers; i++) {
                weightDelta_t[i].clear();
                biasDelta_t[i].clear();
                delta_t[i].clear();
            }
            // Process a minibatch (starts at index j)
            for(int i=0; i<miniBatchSize; i++) {
                uint index = indexList.at(i+j); // Get index
                feedForward(inputSet[index], aoutputs_t, zoutputs_t);
                // Check if correct
                if(checkFunc_t(aoutputs_t[layers-1], outputSet[index]))
                    threadCorrect[ID]++;
                // Compute cost
                cost += getCost(outputSet, aoutputs_t);
                computeOutputError(outputSet[index], aoutputs_t, delta_t, biasDelta_t);
                backPropagate(aoutputs_t, zoutputs_t, delta_t, weightDelta_t, biasDelta_t);
            }
            // Get a mutex to allow use to apply gradient Descent
            taskLock.lock();
            gradientDescent(weightDelta_t, biasDelta_t); // Update weights and biases with the deltas from the batch
            taskLock.unlock();
        }
        // Done training the assigned data
        status[ID] = false;     // Reset status
    }
}

void Network::feedForward(const Matrix<>& input) {
    aoutputs[0] = input;
    for(int i=1; i<layers; i++) {
        add(weights[i]*aoutputs[i-1], biases[i], zoutputs[i]);
        aoutputs[i] = zoutputs[i];
        aoutputs[i].apply(Network::sigmoid); // Apply the sigmoid function to the output
    }
}

// Multithreading friendly version
inline void Network::feedForward(const Matrix<>& input, Matrix<>* aoutputs_t, Matrix<>* zoutputs_t) {
    aoutputs_t[0] = input; // Assume the input is of the right length
    for(int i=1; i<layers; i++) {
        add(weights[i]*aoutputs_t[i-1], biases[i], zoutputs_t[i]); // Note aoutputs[0] = input
        aoutputs_t[i] = zoutputs_t[i];
        aoutputs_t[i].apply(Network::sigmoid); // Apply the sigmoid function to the output
    }
}

void Network::computeOutputError(const Matrix<>& output) {
    // Use cross entropy
    subtract(aoutputs[layers-1], output, delta[layers-1]);
}

// MT friendly version
inline void Network::computeOutputError(const Matrix<> &output, const Matrix<>* aoutputs_t, Matrix<>* delta_t, Matrix<>* biasDelta_t) {
    // Use cross entropy
    subtract(aoutputs_t[layers-1], output, delta_t[layers-1]);
}

void Network::backPropagate() {
    // Calculate deltas for remaining layers
    for (int i=layers-2; i>0; i--) {
        weights[i+1].trans();
        hadamard( weights[i+1]*delta[i+1] , zoutputs[i].apply(dsigmoid, true) , delta[i]);
        weights[i+1].trans(); // Undo transposition
    }
    // Update weight and bias deltas. None are associated with the input layer
    for(int i=1; i<layers; i++) {
        aoutputs[i-1].trans();
        Matrix<> diff = delta[i]*aoutputs[i-1];
        aoutputs[i-1].trans(); // Undo transpose
        weightDelta[i] += diff;
        biasDelta[i] += delta[i];
    }
}

// MT friendly version
inline void Network::backPropagate(Matrix<>* aoutputs_t, Matrix<>* zoutputs_t, Matrix<>* delta_t, Matrix<>* weightDelta_t, Matrix<>* biasDelta_t) {
    // Calculate deltas for remaining layers
    for (int i=layers-2; i>0; i--) {
        // Since (A*B).T = B.T * A.T, weights.T*delta_t = (delta_t.T * weights).T
        // That way we don't ever have to transpose the weights matrix, which leads to
        // problems, since it is asynchronously accessed by all threads
        delta_t[i+1].trans();
        Matrix<> step = delta_t[i+1]*weights[i+1];
        delta_t[i+1].trans();
        step.trans();
        hadamard( step, zoutputs_t[i].apply(dsigmoid, true) , delta_t[i]);
    }
    // Update weight and bias deltas. None are associated with the input layer
    for(int i=1; i<layers; i++) {
        aoutputs_t[i-1].trans();
        Matrix<> diff = delta_t[i]*aoutputs_t[i-1];
        aoutputs_t[i-1].trans(); // Undo transpose
        weightDelta_t[i] += diff;
        biasDelta_t[i] += delta_t[i];
    }
}

void Network::gradientDescent() {
    for(int i=1; i<layers; i++) {
        weights[i] -= L2factor*(1.f/NData)*weights[i]; // L2 regularization
        weights[i] -= (factor*weightDelta[i]);  // Update weights
        biases[i] -= (factor*biasDelta[i]);     // Update biases
    }
}

// MT friendly version
inline void Network::gradientDescent(Matrix<>* weightDelta_t, Matrix<>* biasDelta_t) {
    for(int i=1; i<layers; i++) {
        weights[i] -= L2factor*(1.f/NData)*weights[i]; // L2 regularization
        weights[i] -= (factor*weightDelta_t[i]);  // Update weights
        biases[i] -= (factor*biasDelta_t[i]);     // Update biases
    }
}

float Network::getCost(Matrix<>& expected) {
    float err = 0;
    Matrix<>& output = aoutputs[layers-1];
    for(int i=0; i<sizes[layers-1]; i++) {
        float E = (expected.at(i,0) - output.at(i,0));
        err += E*E;
    }
    err *= 0.5f;
    return err;
}

// MT friendly version
inline float Network::getCost(Matrix<> *expected, Matrix<>* aoutputs_t) {
    float err = 0;
    for(int i=0; i<sizes[layers-1]; i++) {
        float E = (expected[i].at(i,0) - aoutputs_t[layers-1].at(i,0));
        err += E*E;
    }
    err *= 0.5f;
    return err;
}

// Return the output layer of the network
Matrix<> Network::getOutput() {
    return aoutputs[layers-1];
}

Matrix<> Network::getError(uint layer) {
    if (layer<layers) {
        return delta[layer];
    }
    else throw LayerNonexistent();
}

Matrix<>& Network::getWeights(uint layer) {
    if(layer<layers && layer>0) {
        return weights[layer];
    }
    else throw LayerNonexistent();
}

Matrix<>& Network::getBiases(uint layer) {
    if(layer<layers && layer>0) {
        return biases[layer];
    }
    else throw LayerNonexistent();
}

string Network::getCostRecord() {
    stringstream S;
    S << "{";
    for(float num : CostRecord) {
        S << num;
        S << ",";
    }
    string out;
    S >> out;
    out.pop_back();
    out += "}";
    return out;
}

uint Network::getComplexity() {
    uint C = 0;
    for(int i=1; i<layers; i++)
        C += (sizes[i-1]*sizes[i]);
    return C;
}

void Network::setMiniBatchSize(uint size) {
    miniBatchSize = size;
    factor=eta/miniBatchSize;
}

void Network::setEta(float eta) {
    this->eta = eta;
    factor=eta/miniBatchSize;
    L2factor = eta*lambda;
}

void Network::setLambda(float lambda) {
    this->lambda = lambda;
    L2factor = eta*lambda;
}

void Network::popLayer() {
    layers--;
}

void Network::addLayers(uint layers, uint* sizes) {
    uint newTotal = layers + this->layers;
    uint *newSizes = new uint[newTotal];
    for (int i=0; i<this->layers; i++) newSizes[i] = this->sizes[i];
    for (int i=layers; i<newTotal; i++) newSizes[i] = sizes[i-this->layers];
    sizes = newSizes;
        
    // Create weights and biases
    auto newWeights = new Matrix<>[newTotal];       // weights[0] is null
    auto newBiases = new Matrix<>[newTotal];        // biases[0] is null
    
    int i;
    for (i=1; i<layers; i++) { // Initialize random weights
        newWeights[i] = weights[i];
        newBiases[i] = biases[i];
    }
    for (; i<newTotal; i++) {
        cout << sizes[i] << ", " << sizes[i-1] << endl;
        newWeights[i] = Matrix<>::randMatrix(sizes[i], sizes[i-1]);
        newWeights[i] *= 1.f/sqrt(sizes[i-1]);
        newBiases[i] = Matrix<>::randMatrix(sizes[i], 1);
    }
    weights = newWeights;
    biases = newBiases;
    this->layers = newTotal;
    setHelperMatrices();
}
