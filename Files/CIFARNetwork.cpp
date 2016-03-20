//
//  CIFARNetwork.cpp
//  Network
//
//  Created by Nathaniel Rupprecht on 7/3/15.
//  Copyright (c) 2015 Nathaniel Rupprecht. All rights reserved.
//

#include "CIFARNetwork.h"

void CIFARNetwork::run() {
    // Set up network parameters *******/
    uint iter = 30;
    float eta = 0.05f, lambda = 0.f;
    bool etaDecay = false;
    const uint layers = 3;
    uint sizes[layers] = {3072,30,10};
    //**********************************/
    
    Network network(layers, sizes);
    CifarUnpacker unpacker;
    
    cout << "Complexity: " << network.getComplexity() << endl;
    
    // Get training set
    vector<string> fileNames;
    fileNames.push_back("data_batch_1.bin");
    //fileNames.push_back("data_batch_2.bin");
    //fileNames.push_back("data_batch_3.bin");
    //fileNames.push_back("data_batch_4.bin");
    //fileNames.push_back("data_batch_5.bin");
    unpacker.unpackInfo(fileNames);
    vector<float*> inputSet = unpacker.getInputSet();
    vector<float*> outputSet = unpacker.getLabelSet();
    
    // Get testing set
    unpacker.unpackInfo(vector<string>({string("test_batch.bin")}));
    vector<float*> testInput = unpacker.getInputSet();
    vector<float*> testOutput = unpacker.getLabelSet();
    
    // Set network parameters
    network.setTrainingIterations(iter);
    network.setEta(eta);
    network.setLambda(lambda);
    network.setEtaDecay(etaDecay);
    network.setPeriodicSave(true);
    
    network.setNThreads(2);
    network.mtTrainOnDataset(checkFunc, inputSet, outputSet);
    //network.mtTrainOnDataset(checkFunc, inputSet, outputSet, testInput, testOutput);
    
    cout << network.getCostRecord() << endl;
    
    // Test saving the network
    string netName = "N(";
    for(int i=0; i<layers; i++) netName += (toString(sizes[i]) + "_");
    netName += (")E" + toString(eta));
    netName += ("_L" + toString(lambda));
    netName += ("_I_" + toString(iter));
    netName += ".net";
    network.saveNetwork(netName);
}

void CIFARNetwork::update(string fileName) {
    // Set up network parameters *******/
    uint iter = 200;
    float eta = 0.05f;
    float lambda = 0.f;
    bool etaDecay = false;
    //**********************************/
    
    Network network;
    CifarUnpacker unpacker;
    network.loadNetwork(fileName);
    
    cout << "Complexity: " << network.getComplexity() << endl;
    
    // Get training set
    vector<string> fileNames;
    fileNames.push_back("data_batch_1.bin");
    fileNames.push_back("data_batch_2.bin");
    fileNames.push_back("data_batch_3.bin");
    fileNames.push_back("data_batch_4.bin");
    fileNames.push_back("data_batch_5.bin");
    unpacker.unpackInfo(fileNames);
    vector<float*> inputSet = unpacker.getInputSet();
    vector<float*> outputSet = unpacker.getLabelSet();
    
    // Get testing set
    unpacker.unpackInfo(vector<string>({string("test_batch.bin")}));
    vector<float*> testInput = unpacker.getInputSet();
    vector<float*> testOutput = unpacker.getLabelSet();
    
    // Set network parameters
    network.setTrainingIterations(iter);
    network.setEta(eta);
    network.setLambda(lambda);
    network.setEtaDecay(etaDecay);
    network.setPeriodicSave(true, 1); // Save after each iteration
    
    network.setNThreads(3);
    network.mtTrainOnDataset(checkFunc, inputSet, outputSet); // No testing data for now
    //network.mtTrainOnDataset(checkFunc, inputSet, outputSet, testInput, testOutput);
    
    cout << network.getCostRecord() << endl;
    
    // Test saving the network
    string netName = "N";
    for(int i=0; i<network.getLayers(); i++) netName += (toString(network.getSizes()[i]) + "_");
    netName += ("E_" + toString(eta));
    netName += ("L_" + toString(lambda));
    netName += ("I_" + toString(iter));
    netName += ".net";
    network.saveNetwork(netName);
}

bool CIFARNetwork::checkFunc(const Matrix<>& guess, const Matrix<>& result) {
    
    return getMaxIndex(guess).first == getMaxIndex(result).first;
}