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
#include "Tensor.h"

typedef unsigned int uint;
typedef unsigned char uchar;

class FileUnpack {
 public:
  FileUnpack() {};
  FileUnpack(string imageFileName, string labelFileName);
    
  void unpackInfo();
    
  BMP getImage(uint index);
  Tensor& getLabel(uint index);
    
  // Accessors
  vector<Tensor*> getImages() {return images;}
  vector<Tensor*> getLabels() {return labels;}
    
  // Mutators
  void setFileNames(string image, string label);
    
 private:
    
  uint getI32(ifstream& fin);
    
  string imageFileName;
  string labelFileName;
    
  // Unpacked info vectors
  vector<Tensor*> images;  // Vector of images
  vector<Tensor*> labels;   // Vector of labels
};

#endif // MNIST_UNPACK_H
