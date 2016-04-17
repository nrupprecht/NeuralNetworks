// Test

#include "Tensor.h"
#include "Matrix.h"

#include <iostream>
using std::cout;
using std::endl;

int main(int argc, char* argv[]) {

  Shape mat(5,5), vec(5);
  Tensor Mat(mat), Vec(vec), Vec2(vec);
  
  Mat.random(); Vec.random();

  multiply(Mat, Vec, Vec2);

  cout << "mat=" << Mat << ";\n";
  cout << "vec=" << Vec << ";\n";
  cout << "vec2=" << Vec2 << ";\n";

  return 0;
  

  clock_t start, end;

  int n=1000, m=1212, k=987;
  Shape s(n, k), t(k,m), u(n,m);
  Tensor S(s), T(t), U(u);
  Matrix M(n,k), N(k,m), P(n,m);
  
  S.random(); T.random();

  start = clock();  
  multiply(S, 1, T, 0, U);
  end = clock();
  cout << "Time: " << static_cast<float>(end - start)/CLOCKS_PER_SEC << endl;

  start = clock();
  multiply(M, N, P);
  end = clock();
  cout << "Time: " << static_cast<float>(end - start)/CLOCKS_PER_SEC << endl;

  return 0;
}
