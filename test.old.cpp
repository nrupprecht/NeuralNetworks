#include <iostream>
#include <stdlib.h> // For drand48
#include <stdio.h>
#include <ctime> // For timing


#include "/afs/crc.nd.edu/x86_64_linux/intel/15.0/mkl/include/mkl.h"
//#include "/afs/crc.nd.edu/x86_64_linux/intel/15.0/mkl/include/mkl_blas.h"

#include <string>
#include <sstream>

using std::cout;
using std::endl;
using std::string;

void square_dgemm (int N, double* A, double* B, double* C)
{
  char TRANSA = 'N';
  char TRANSB = 'N';
  int M = N;
  int K = N;
  double ALPHA = 1.;
  double BETA = 1.;
  int LDA = N;
  int LDB = N;
  int LDC = N;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, ALPHA, A, N, B, N, BETA, C, N);
}

string print(double *C, int N) {
  std::stringstream stream;
  stream << "{";
  for (int i=0; i<N; i++) {
    stream << "{";
    for (int j=0; j<N; j++) {
      stream << C[i*N+j];
      if (j!=N-1) stream << ",";
    }
    stream << "}";
    if (i!=N-1) stream <<",";
  }
  stream << "}";
  string str;
  stream >> str;
  return str;
}

int main() {
  int N = 3000;
  double *a = new double[N*N];
  double *b = new double[N*N];
  double *c = new double[N*N];

  for (int i=0; i<N*N; i++) a[i] = 1+drand48();
  for (int i=0; i<N*N; i++) b[i] = 1+drand48();

  clock_t start = clock();  
  square_dgemm(N, a, b, c);
  clock_t end = clock();
  cout << "Program took " << static_cast<float>(end - start)/CLOCKS_PER_SEC << " seconds." << endl;

  //cout << "A=" << print (a,N) << ";" << endl;
  //cout << "B=" << print (b,N) << ";" << endl;
  //cout << "C=" << print (c,N) << ";" << endl;

  return 0;
}
