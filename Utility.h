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

#endif
