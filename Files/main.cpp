//
//  main.cpp
//  Network
//
//  Created by Nathaniel Rupprecht on 6/27/15.
//  Copyright (c) 2015 Nathaniel Rupprecht. All rights reserved.
//

#include "MINSTNetwork.h"
#include "CIFARNetwork.h"

bool checkFunc(const Matrix<>& in, const Matrix<>& out) {
    if((in.at(0,0) > in.at(1,0) && out.at(0,0) > out.at(1,0)) || (in.at(0,0) < in.at(1,0) && out.at(0,0) < out.at(1,0)))
        return true;
    return false;
}

int main(int argc, const char * argv[]) {

    MINSTNetwork net;
    net.run();
    //net.autoEncRun();
    
    //CIFARNetwork net;
    //net.run();
    
    return 0;
}
