//
//  MINSTNetwork.h
//  Network
//
//  Created by Nathaniel Rupprecht on 7/1/15.
//  Copyright (c) 2015 Nathaniel Rupprecht. All rights reserved.
//

#ifndef __Network__MINSTNetwork__
#define __Network__MINSTNetwork__

#include "Network.h"
#include "FileUnpack.h"

class MINSTNetwork {
public:
    void run();
    
    void autoEncRun();
    
    void update(string fileName);
    void updateAndAdd(string fileName);
    
private:
    
    static bool checkFunc(const Matrix<>& guess, const Matrix<>& result);
    
    std::pair<vector<float*>, vector<float*> > convertForm(vector<uchar*>, vector<uchar>);
};

#endif /* defined(__Network__MINSTNetwork__) */
