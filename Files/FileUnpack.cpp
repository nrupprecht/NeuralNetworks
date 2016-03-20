//
//  FileUnpack.cpp
//  FileUnpack
//
//  Created by Nathaniel Rupprecht on 6/29/15.
//  Copyright (c) 2015 Nathaniel Rupprecht. All rights reserved.
//

#include "FileUnpack.h"

#include <stdint.h>

#include <iostream>
using std::cout;
using std::endl;

FileUnpack::FileUnpack(string imageFileName, string labelFileName) {
    this->imageFileName = imageFileName;
    this->labelFileName = labelFileName;
}

void FileUnpack::unpackInfo() {
    ifstream ifin(imageFileName);
    if(ifin.fail()) {
        cout << "File " << imageFileName << " failed to open." << endl;
        return;
    }
    ifstream lfin(labelFileName);
    if(lfin.fail()) {
        cout << "File " << labelFileName << " failed to open." << endl;
        return;
    }
    
    int magicnum, rows, cols;
    
    // First, unpack images
    // Get the first four numbers
    magicnum = getI32(ifin);    // Get the magic number
    getI32(ifin);      // Try to get the number of items
    rows = getI32(ifin);
    cols = getI32(ifin);
    
    if(magicnum != 2051) {
        // Not really the image file
        throw 2;
    }
    
    int bytes = rows*cols;
    char ubyte; // An unsigned byte
    
    while(!ifin.eof()) { // Read to the end of the file
        uchar *array = new uchar[bytes];
        for(int i=0; i<bytes; i++) {
            ifin.get(ubyte);
            array[i] = ubyte;
        }
        images.push_back(array);
    } // EOF
    images.pop_back(); // The last "image" isn't really an image
    
    // Now, unpack labels
    magicnum = getI32(lfin); // Get the magic number
    if(magicnum != 2049) {
        // Not really the label file
        throw 2;
    }
    getI32(lfin); // Try to get the number of items
    while(!lfin.eof()) {
        lfin.get(ubyte);
        labels.push_back(ubyte);
    }
    labels.pop_back(); // The last label isn't really a label
}

uint FileUnpack::getI32(ifstream& fin) {
    char n1, n2, n3, n4;
    fin >> n1;
    fin >> n2;
    fin >> n3;
    fin >> n4;
    
    return (uint)n4 + (((uint)n3)<<8) + (((uint)n2) << 16) + (((uint)n1) << 24);
}

BMP FileUnpack::getImage(uint index) {
    BMP image;
    image.SetSize(28, 28);
    
    for(int y=0; y<28; y++)
        for(int x=0; x<28; x++) {
            int col = 255 - images.at(index)[28*y+x];
            image.SetPixel(x, y, RGBApixel(col, col, col));
        }
    
    return image;
}

uint FileUnpack::getLabel(uint index) {
    return labels.at(index);
}

void FileUnpack::setFileNames(string image, string label) {
    imageFileName=image;
    labelFileName=label;
    images.clear();
    labels.clear();
}
