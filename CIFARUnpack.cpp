//
//  Cifar.cpp
//  Network
//
//  Created by Nathaniel Rupprecht on 7/3/15.
//  Copyright (c) 2015 Nathaniel Rupprecht. All rights reserved.
//

#include "CIFARUnpack.h"

#include <fstream>
using std::ifstream;

#include <iostream>
using std::cout;
using std::endl;

void CifarUnpacker::unpackInfo(vector<string> fileNames) {

    // There should be
    for(auto name : fileNames) {
        ifstream fin(name);
        if(fin.fail()) {
            cout << "File " << name << " failed to open.";
            continue;
        }
        // 10000 Images per file
        for(int i=0; i<10000; i++) {
            // Get label
            char c;
            fin.get(c);
            uint label = (uint)c;
            // Convert label data
            Matrix *M = new Matrix(10,1);
	    for(int p=0; p<10; p++) M->at(p,0) = p==label ? 1. : 0.;
            labels.push_back(M);
            
	    Matrix *pixels = new Matrix(3072,1);
            // Get pixels
            int j;
            for(j=0; j<3072 && !fin.eof(); j++) { //1024 for Red,Green,Blue -> 3072
                fin.get(c);
                pixels->at(j,0) = (double)((unsigned char)c)/255.f;
            }
            images.push_back(pixels);
        }
    }
}

