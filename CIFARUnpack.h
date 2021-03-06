//
//  Cifar.h
//  Network
//
//  Created by Nathaniel Rupprecht on 7/3/15.
//  Copyright (c) 2015 Nathaniel Rupprecht. All rights reserved.
//

#ifndef __Network__Cifar__
#define __Network__Cifar__

#include <stdio.h>
#include <string>
using std::string;
#include <vector>
using std::vector;

#include "Tensor.h"

// Store image as a Matrix (column vector) [ red green blue ]
class CifarUnpacker {
public:
    CifarUnpacker() {};
    
    void unpackInfo(vector<string> fileNames);
    
    vector<Tensor*> getInputSet()  {return images;}
    vector<Tensor*>& getLabelSet() {return labels;}
    
private:
    
    // Store the information
    vector<Tensor*> images;
    vector<Tensor*> labels;
};

#endif /* defined(__Network__Cifar__) */
