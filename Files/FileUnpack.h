//
//  FileUnpack.h
//  FileUnpack
//
//  Created by Nathaniel Rupprecht on 6/29/15.
//  Copyright (c) 2015 Nathaniel Rupprecht. All rights reserved.
//

#ifndef __FileUnpack__FileUnpack__
#define __FileUnpack__FileUnpack__

#include <stdio.h>

#include <string>
using std::string;

#include <fstream>
using std::ifstream;

#include <vector>
using std::vector;

#include "EasyBMP.h"

typedef unsigned int uint;
typedef unsigned char uchar;

class FileUnpack {
public:
    FileUnpack() {};
    FileUnpack(string imageFileName, string labelFileName);
    
    void unpackInfo();
    
    BMP getImage(uint index);
    uint getLabel(uint index);
    
    // Accessors
    vector<uchar*> getImages() {return images;}
    vector<uchar> getLabels() {return labels;}
    
    // Mutators
    void setFileNames(string image, string label);
    
private:
    
    uint getI32(ifstream& fin);
    
    string imageFileName;
    string labelFileName;
    
    // Unpacked info vectors
    vector<uchar*> images;  // Vector of images
    vector<uchar> labels;   // Vector of labels
};

#endif /* defined(__FileUnpack__FileUnpack__) */
