//
//  CIFARNetwork.h
//  Network
//
//  Created by Nathaniel Rupprecht on 7/3/15.
//  Copyright (c) 2015 Nathaniel Rupprecht. All rights reserved.
//

#ifndef __Network__CIFARNetwork__
#define __Network__CIFARNetwork__

#include <stdio.h>

#include "Cifar.h"
#include "Network.h"

class CIFARNetwork {
public:
    void run();
    
    void update(string fileName);
    
private:
    static bool checkFunc(const Matrix<>& guess, const Matrix<>& result);
};

#endif /* defined(__Network__CIFARNetwork__) */
