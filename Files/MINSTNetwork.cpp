//
//  MINSTNetwork.cpp
//  Network
//
//  Created by Nathaniel Rupprecht on 7/1/15.
//  Copyright (c) 2015 Nathaniel Rupprecht. All rights reserved.
//

#include "MINSTNetwork.h"

void MINSTNetwork::run() {
    // Set up network parameters *******//
    uint iter = 100;
    uint minibatch = 2;
    float eta = 0.5f, lambda = 0.f;
    bool etaDecay = false;
    const uint layers = 3;
    uint sizes[layers] = {784,30,10};
    //**********************************//
    
    Network network(layers, sizes);
    
    cout << "Complexity: " << network.getComplexity() << endl;
    
    FileUnpack unpacker("../trainingImages","../trainingLabels");
    
    vector<float*> inputVectors;
    vector<float*> outputVectors;
    
    // Set up input
    unpacker.unpackInfo();
    cout << endl;
    auto rawInput = unpacker.getImages();
    auto rawOutput = unpacker.getLabels();
    auto P = convertForm(rawInput, rawOutput);
    inputVectors = P.first;
    outputVectors = P.second;
    
    unpacker.setFileNames("../testImages", "../testLabels");
    unpacker.unpackInfo();
    cout << endl;
    vector<uchar*> testInput = unpacker.getImages();
    vector<uchar> testOutput = unpacker.getLabels();
    std::pair<vector<float*>,vector<float*> > Q = convertForm(testInput, testOutput);
    vector<float*> testInVectors = Q.first;
    vector<float*> testOutVectors = Q.second;
    
    network.setTrainingIterations(iter);
    network.setEta(eta);
    network.setLambda(lambda);
    network.setEtaDecay(etaDecay);
    network.setMiniBatchSize(minibatch);
    network.setNThreads(2);
    //network.setPeriodicSave(true);
    
    network.trainOnDataset(checkFunc, inputVectors, outputVectors, testInVectors, testOutVectors);
    //network.mtTrainOnDataset(checkFunc, inputVectors, outputVectors, testInVectors, testOutVectors);
    
    cout << network.getCostRecord() << endl;
    
    // Test saving the network
    if ((false)) {
        string netName = "N";
        for(int i=0; i<layers; i++) netName += (toString(sizes[i]) + "_");
        netName += ("E_" + toString(eta));
        netName += ("L_" + toString(lambda));
        netName += ("I_" + toString(iter));
        netName += ".net";
        network.saveNetwork(netName);
    }
}

void MINSTNetwork::autoEncRun() {
    // Set up network parameters *******//
    uint iter = 100;
    uint minibatch = 10;
    float eta = 0.5f, lambda = 0.f;
    bool etaDecay = true;
    const uint layers = 3;
    uint sizes[layers] = {784,30,784};
    //**********************************//
    
    Network network(layers, sizes);
    
    cout << "Complexity: " << network.getComplexity() << endl;
    
    FileUnpack unpacker("../trainingImages","../trainingLabels");
    
    vector<float*> inputVectors;
    vector<float*> outputVectors;
    
    // Set up input
    unpacker.unpackInfo();
    cout << endl;
    auto rawInput = unpacker.getImages();
    auto rawOutput = unpacker.getLabels();
    auto P = convertForm(rawInput, rawOutput);
    inputVectors = P.first;
    outputVectors = P.second;
    
    unpacker.setFileNames("../testImages", "../testLabels");
    unpacker.unpackInfo();
    cout << endl;
    vector<uchar*> testInput = unpacker.getImages();
    vector<uchar> testOutput = unpacker.getLabels();
    std::pair<vector<float*>,vector<float*> > Q = convertForm(testInput, testOutput);
    vector<float*> testInVectors = Q.first;
    vector<float*> testOutVectors = Q.second;
    
    // Set parameters
    network.setEta(eta);
    network.setLambda(lambda);
    network.setEtaDecay(etaDecay);
    network.setMiniBatchSize(minibatch);
    
    // Auto encoder step
    network.setTrainingIterations(0);
    network.autoEncoder(inputVectors);
    
    // Reset some parameters
    network.setTrainingIterations(iter);
    network.setPeriodicSave(true);
    
    network.popLayer();
    uint addSizes[2] = {50, 10};
    network.addLayers(2, addSizes);
    network.trainOnDataset(checkFunc, inputVectors, outputVectors, testInVectors, testOutVectors);
    
    cout << network.getCostRecord() << endl;
    
    // Test saving the network
    string netName = "N";
    for(int i=0; i<layers; i++) netName += (toString(sizes[i]) + "_");
    netName += ("E_" + toString(eta));
    netName += ("L_" + toString(lambda));
    netName += ("I_" + toString(iter));
    netName += ".net";
    network.saveNetwork(netName);

}

void MINSTNetwork::update(string fileName) {
    // Set up network parameters *******/
    uint iter = 30;
    float eta = 0.25, lambda = 0.f;
    //**********************************/
    
    Network network;
    network.loadNetwork(fileName);
    
    cout << "Complexity: " << network.getComplexity() << endl;
    
    FileUnpack unpacker("trainingimages","traininglabels");
    
    vector<float*> inputVectors;
    vector<float*> outputVectors;
    
    // Set up input
    unpacker.unpackInfo();
    cout << endl;
    auto rawInput = unpacker.getImages();
    auto rawOutput = unpacker.getLabels();
    auto P = convertForm(rawInput, rawOutput);
    inputVectors = P.first;
    outputVectors = P.second;
    // Get data
    unpacker.setFileNames("testImages", "testLabels");
    unpacker.unpackInfo();
    cout << endl;
    vector<uchar*> testInput = unpacker.getImages();
    vector<uchar> testOutput = unpacker.getLabels();
    std::pair<vector<float*>,vector<float*> > Q = convertForm(testInput, testOutput);
    vector<float*> testInVectors = Q.first;
    vector<float*> testOutVectors = Q.second;
    
    network.setTrainingIterations(iter);
    network.setEta(eta);
    network.setLambda(lambda);
    
    network.trainOnDataset(checkFunc, inputVectors, outputVectors, testInVectors, testOutVectors);
    
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

void MINSTNetwork::updateAndAdd(string fileName) {
    // Set up network parameters *******/
    uint iter = 30;
    float eta = 0.25, lambda = 0.f;
    //**********************************/
    
    Network network;
    network.loadNetworkAndAdd(fileName, 30);
    
    FileUnpack unpacker("trainingimages","traininglabels");
    
    vector<float*> inputVectors;
    vector<float*> outputVectors;
    
    // Set up input
    unpacker.unpackInfo();
    cout << endl;
    auto rawInput = unpacker.getImages();
    auto rawOutput = unpacker.getLabels();
    auto P = convertForm(rawInput, rawOutput);
    inputVectors = P.first;
    outputVectors = P.second;
    
    unpacker.setFileNames("testImages", "testLabels");
    unpacker.unpackInfo();
    cout << endl;
    vector<uchar*> testInput = unpacker.getImages();
    vector<uchar> testOutput = unpacker.getLabels();
    std::pair<vector<float*>,vector<float*> > Q = convertForm(testInput, testOutput);
    vector<float*> testInVectors = Q.first;
    vector<float*> testOutVectors = Q.second;
    
    network.setTrainingIterations(iter);
    network.setEta(eta);
    network.setLambda(lambda);
    
    network.trainOnDataset(checkFunc, inputVectors, outputVectors, testInVectors, testOutVectors);
    
    cout << network.getCostRecord() << endl;
}

bool MINSTNetwork::checkFunc(const Matrix<> &guess, const Matrix<> &result) {
    return getMaxIndex(guess).first == getMaxIndex(result).first;
}

std::pair<vector<float*>, vector<float*> > MINSTNetwork::convertForm(vector<uchar*> images, vector<uchar> labels) {
    vector<float*> inputVectors;
    vector<float*> outputVectors;
    
    for(int i=0; i<images.size(); i++) {
        // Convert image data
        float *array = new float[784];
        for(int p=0; p<784; p++) array[p] = (float)images.at(i)[p]/255.f; // Not making 255 a float led to big problems
        inputVectors.push_back(array);
        
        // Convert label data
        float *labelarray = new float[10];
        for(int p=0; p<10; p++) {
            if(p == (int)labels.at(i))
                labelarray[p]=1.f;
            else labelarray[p] = 0.f;
        }
        outputVectors.push_back(labelarray);
    }
    return std::pair<vector<float*>,vector<float*> >(inputVectors, outputVectors);
}
