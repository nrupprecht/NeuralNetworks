// MNISTUnpck.h
//  Origionally created Nathaniel Rupprecht 6/29/15.
//

#ifndef MNIST_UNPACK_H
#define MNIST_UNPACK_H

#include <stdio.h>
#include <string>
using std::string;
#include <fstream>
using std::ifstream;
#include <vector>
using std::vector;

#include "EasyBMP/EasyBMP.h"
#include "Matrix.h"

typedef unsigned int uint;
typedef unsigned char uchar;

class FileUnpack {
 public:
  FileUnpack() {};
  FileUnpack(string imageFileName, string labelFileName);
    
  void unpackInfo();
    
  BMP getImage(uint index);
  Matrix& getLabel(uint index);
    
  // Accessors
  vector<Matrix*> getImages() {return images;}
  vector<Matrix*> getLabels() {return labels;}
    
  // Mutators
  void setFileNames(string image, string label);
    
 private:
    
  uint getI32(ifstream& fin);
    
  string imageFileName;
  string labelFileName;
    
  // Unpacked info vectors
  vector<Matrix*> images;  // Vector of images
  vector<Matrix*> labels;   // Vector of labels
};

#endif // MNIST_UNPACK_H
