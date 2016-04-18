/// Shape.h
/// Nathaniel Rupprecht
///

#ifndef SHAPE_H
#define SHAPE_H

#include "Utility.h"

struct Shape {
  Shape() : rank(0), total(0), dims(0) {};
  
  template<typename ... T> Shape(int first, T ...s) {
    vector<int> vect;
    unpack(vect, first, s...);
    rank = vect.size();
    if (rank > 0) {
      dims = new int[rank];
      total = 1;
      for (int i=0; i<rank; i++) {
        dims[i] = vect.at(i);
        total *= dims[i];
      }
    }
    else {
      dims = 0;
      total = 0;
    }
  }

  Shape(const Shape& s) : dims(0) { *this = s; }

  Shape& operator=(const Shape& s) {
    // Copy data, not pointers
    rank = s.rank;
    total = s.total;
    if (dims) delete [] dims;
    dims = new int[rank];
    for (int i=0; i<rank; i++) dims[i] = s.dims[i];
    return *this;
  }
    
  ~Shape() { 
    if(dims) delete [] dims; 
  }

  friend Shape operator+(const Shape& A, const Shape& B) {
    int rank = A.rank+B.rank;
    Shape S;
    S.rank = rank;
    int i;
    // Set dims
    S.dims = new int[rank];
    for (i=0; i<A.rank; i++) S.dims[i] = A.dims[i];
    for (int k=0; i<rank; i++, k++) S.dims[i] = B.dims[k];
    // Compute total
    int total = 1;
    for (i=0; i<rank; i++) total *= S.dims[i];
    S.total = total;
    // Return
    return S;
  }

  friend ostream& operator<<(ostream& out, const Shape& s) {
    if (s.rank==0) {
      out << "{}";
    }
    else {
      out << "{";
      for (int i=0; i<s.rank; i++) {
	out << s.dims[i];
	if (i!=s.rank-1) out << ",";
      }
      out << "}";
    }
    return out;
  }

  bool operator==(const Shape& S) const {
    if (rank!=S.rank) return false;
    for (int i=0; i<rank; i++)
      if (S.dims[i]!=dims[i]) return false;
    return true;
  }

  int getTotal() const { return total; }

  int at(int i) const {
    if (i<0 || i>=rank) throw ShapeOutOfBounds();
    return dims[i];
  }
  
  // Error classes
  class ShapeOutOfBounds {};

  int rank;
  int* dims;

private:
  // Helper functions
  static void unpack(vector<int>&) {};
  template <typename ... T> static void unpack(vector<int>& vect, int first, T ... last) {
    vect.push_back(first);
    unpack(vect, last...);
  }

  int total;
};

#endif
