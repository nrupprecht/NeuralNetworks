#ifndef UTILITY_H
#define UTILITY_H

/// For use with the CRC
#include"/afs/crc.nd.edu/x86_64_linux/intel/15.0/mkl/include/mkl.h"

#include <math.h>   // For sqrt
#include <stdlib.h> // For drand48
#include <ctime>    // For timing

#include <vector>   // For vector
using std::vector;

#include <string>   // For string
using std::string;

#include <sstream>  // For string stream
using std::stringstream;

#include <ostream>
using std::ostream;
#include <fstream>  // For file i/o

// For debugging
#include <iostream>
using std::cout;
using std::endl;

// Common function template
typedef double (*function) (double);

// Min/Max functions
template<typename T> T min(T a, T b) { return a<b?a:b; }
template<typename T> T max(T a, T b) { return a<b?b:a; }

template<typename T> string print(const vector<T>& array) {
  if (array.size()==0) return "{}";
  string str;
  stringstream stream;
  stream << "{";
  for (int i=0; i<array.size()-1; i++) stream << array.at(i) << ",";
  stream << array.at(array.size()-1) << "}";
  stream >> str;
  return str;
}

#endif
