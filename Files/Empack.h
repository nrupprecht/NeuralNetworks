//
//  Empack.h
//
//  Created by Nathaniel Rupprecht on 7/8/15.
//  Copyright (c) 2015 Nathaniel Rupprecht. All rights reserved.
//

#ifndef __Empack__Matrix__
#define __Empack__Matrix__

#include <cstdlib>
#include <stdio.h>
#include <random>
#include <string>
#include <sstream>
#include <iostream>
using std::cout;
using std::endl;

// Typedefs
typedef unsigned int uint;

// Foreward declaration
template<typename T> class Matrix;

// Global variable
const uint BLOCKSIZE = 16;

enum class MatrixType { RANDOM, IDENTITY, ZERO };

// Random engines
static std::default_random_engine generator;
static std::normal_distribution<float> floatDist(0, 1.f);
static std::uniform_real_distribution<float> ufloatDist(0.005f,0.995f);
static std::uniform_int_distribution<int> intDist(-10,10);

// Forward declarations
template <typename T> class Matrix;
template <typename T> inline void writeToStream(const Matrix<T>& M, std::ostream& out);

//*****************************************************************//
/// Matrix class
class Matrix {
public:
    
    // Exception classes
    class MatrixMismatchError {};
    class MatrixOutOfBounds {};
    
    // Constructors
    Matrix() {
        width = 0;
        height = 0;
        total = 0;
        transposed = false;
        values = 0;
    }
    Matrix(uint rows, uint cols) {
        width = cols;
        height = rows;
        total = width*height;
        transposed = false;
        values = new double[total];
        for(int i=0; i<total; i++) values[i] = 0;
    }
    Matrix(uint rows, uint cols, MatrixType type) {
        width = cols;
        height = rows;
        total = width*height;
        transposed = false;
        values = new double[total];
        switch(type) {
            case MatrixType::RANDOM:
	      for(int i=0; i<total; i++) values[i] = 1+drand48();
                break;
            case MatrixType::IDENTITY: {
                for(int i=0; i<total; i++) values[i] = 0;
                int min = rows<cols ? rows : cols;
                for (int i=0; i<min; i++) at(i,i) = 1;
                break;
            }
            case MatrixType::ZERO:
            default:
	      for(int i=0; i<total; i++) values[i] = 0;
                break;
        }
        
    }
    Matrix(const std::vector<double> vecin) { // Create a vector from a (c++) vector
        total = vecin.size();
        width = 1;
        height = total;
        transposed = false;
        values = new double[total];
        for (int i=0; i<total; i++) values[i] = vecin.at(i);
    }
    Matrix(const double* array, uint length) { // Create a vector from an array
        total = length;
        width = 1;
        height = length;
        transposed = false;
        // Copy values
        values = new double[length];
        for (int i=0; i<length; i++) values[i] = array[i];
    }
    Matrix(const Matrix<T> &M) {
        width = M.width;
        height = M.height;
        total = M.total;
        transposed = M.transposed;
        rm = M.rm;
        // Copy values
        values = new T[total];
        for (int i=0; i<total; i++) values[i] = M.values[i];
    }
    Matrix(Matrix<T> &&M) {
        width = M.width;
        height = M.height;
        total = M.total;
        transposed = M.transposed;
        rm = M.rm;
        values = M.values;
        M.values = 0;
    }
    ~Matrix() {
        if (values) {
            delete [] values;
            values = 0;
            width = height = total = 0; // This isn't actually necessary
        }
    }
    
    // Equals operators
    Matrix operator=(const Matrix<T> &M) {
        if (total == M.total) {
            width = M.width;
            height = M.height;
            rm = M.rm;
            transposed = M.transposed;
            for (int i=0; i<total; i++) values[i] = M.values[i];
        }
        else {
            width = M.width;
            height = M.height;
            total = M.total;
            rm = M.rm;
            transposed = M.transposed;
            if (values) delete [] values;
            values = new T[total];
            for (int i=0; i<total; i++) values[i] = M.values[i];
        }
        return *this;
    }
    
    Matrix operator=(Matrix<T> &&M) {
        width = M.width;
        height = M.height;
        total = M.total;
        rm = M.rm;
        transposed = M.transposed;
        if (values) delete [] values;
        values = M.values;
        M.values = 0;
        return *this;
    }
    
    // Stream operator
    friend std::ostream& operator<<(std::ostream& out, const Matrix<T>& M) {
        writeToStream(M, out);
        return out;
    }
    
    // Arithmetic operations
    void friend multiply(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
        // Check dimensions
        if (A.getWidth() != B.getHeight() || A.getHeight() != C.getHeight() || B.getWidth() != C.getWidth())
            throw MatrixMismatchError();
        // Helping variables
        uint dim = A.getWidth();
        uint width = B.getWidth();
        uint height = A.getHeight();
        uint widthA = A.getWidth();
        uint heightB = B.getHeight();
        
        // TODO: Try some blocking techniques
        
        if(A.rmnt() && !B.rmnt()) {
            for (uint x=0; x<width; x++)
                for (uint y=0; y<height; y++) {
                    T num = 0;
                    for (uint i=0; i<dim; i++) {
                        //num += A.rmAt(y,i)*B.cmAt(i,x);
                        num += A.get(y,i)*B.get(i,x);
                    }
                    C.at(y,x) = num;
                }
        }
        else if(A.rmnt() && B.rmnt()) {
            // Create an array representing a rmnt false layout of B
            // creating and moving the values is much, much faster then
            // using the memory pattern of the matrix as is
            T* array __attribute__(( aligned(sizeof(T)) )) = new T[B.total];

            for(int i=0; i<B.total; i++) {
                uint index = (i/width)+(i%width)*heightB;
                array[index] = B.values[i];
            }
            
            // Now multiply A times our rmnt false copy of B
            for (uint x=0; x<width; x++)
                for (uint y=0; y<height; y++) {
                    T num = 0;
                    for (uint i=0; i<dim; i++)
                        num += A.get(y,i)*array[x*dim+i];
                    C.at(y,x) = num;
                }
            delete [] array;
        }
        else if(!A.rmnt() && B.rmnt()) {
            // Create two arrays representing a rmnt false layout of B
            // and a rmnt true layout of A.
            // Creating and moving the values is much, much faster then
            // using the memory pattern of the matrix as is
            
            //cout << mathematicaForm(A) << endl;
            //cout << mathematicaForm(B) << endl;
            
            T *arrayA __attribute__(( aligned(sizeof(T)) )) = new T[A.total];
            T *arrayB __attribute__(( aligned(sizeof(T)) )) = new T[B.total];
            for(int i=0; i<A.total; i++) {
                uint indexA = (i/height)+(i%height)*widthA;
                arrayA[indexA] = A.values[i];
            }
            for(int i=0; i<B.total; i++) {
                uint indexB = (i/width)+(i%width)*heightB;
                arrayB[indexB] = B.values[i];
            }
            // Now multiply
            for (uint x=0; x<width; x++)
                for (uint y=0; y<height; y++) {
                    T num = 0;
                    for (uint i=0; i<dim; i++)
                        num += arrayA[y*dim+i]*arrayB[x*dim+i];
                    C.at(y,x) = num;
                }
            delete [] arrayA;
            delete [] arrayB;
        }
        else { // !A.rmnt() && !B.rmnt()
            // Create an array representing a rmnt true layout of A
            // creating and moving the values is much, much faster then
            // using the memory pattern of the matrix as is
            T *array __attribute__(( aligned(sizeof(T)) )) = new T[A.total];
            for(int i=0; i<A.total; i++) {
                uint indexA = (i/height)+(i%height)*widthA;
                array[indexA] = A.values[i];
            }
            // Now multiply
            for (uint x=0; x<width; x++)
                for (uint y=0; y<height; y++) {
                    T num = 0;
                    for (uint i=0; i<dim; i++)
                        num += array[y*dim+i]*B.get(i,x);
                    C.at(y,x) = num;
                }
            delete [] array;
        }
        
    }
    
    Matrix operator*(const Matrix<T>& B) {
        Matrix C(getHeight(), B.getWidth());
        multiply(*this, B, C);
        return C;
    }
    
    Matrix operator*=(float m) {
        for(int i=0; i<total; i++) values[i] *= m;
        return *this;
    }
    
    friend Matrix operator*(float m, const Matrix& N) {
        Matrix<T> R;
        R.width = N.getWidth();
        R.height = N.getHeight();
        R.rm = true;
        R.transposed = false;
        R.total = N.total;
        R.values = new T[R.total];
        for(int i=0; i<N.total; i++) R.values[i] = m*N.values[i];
        return R;
    }
    
    void friend add(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
        if(A.getWidth() != B.getWidth() || A.getHeight() != B.getHeight()
           || A.getWidth() != C.getWidth() || A.getHeight() != C.getHeight())
            throw MatrixMismatchError();
        
        uint total = A.total, height = A.height, width = A.width;
        if(A.rmnt() == B.rmnt()) {
            C.rm = A.rm;
            C.transposed = A.transposed;
            for(int i=0; i<total; i++)
                C.values[i] = A.values[i]+B.values[i];
        }
        else {
            for(int y=0; y<height; y++)
                for(int x=0; x<width; x++)
                    C.at(y,x) = A.at(y,x)+B.at(y,x);
        }
    }
    
    Matrix operator+(const Matrix<T>& B) const {
        Matrix C(B.getHeight(), B.getWidth());
        add(*this, B, C);
        return C;
    }
    
    Matrix operator+=(const Matrix<T>& B) {
        if(getWidth() != B.getWidth() || getHeight() != B.getHeight())
            throw MatrixMismatchError();
        
        uint h = getHeight(); uint w = getWidth();
        if(rmnt() == B.rmnt()) {
            for(int i=0; i<total; i++)
                values[i] += B.values[i];
        }
        else {
            for(int y=0; y<h; y++)
                for(int x=0; x<w; x++)
                    at(y,x) += B.at(y,x);
        }
        return *this;
    }
    
    void friend subtract(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
        if(A.getWidth() != B.getWidth() || A.getHeight() != B.getHeight()
           || A.getWidth() != C.getWidth() || A.getHeight() != C.getHeight())
            throw MatrixMismatchError();
            
            uint total = A.total, height = A.height, width = A.width;
        if(A.rmnt() == B.rmnt()) {
            C.rm = A.rm;
            C.transposed = A.transposed;
            for(int i=0; i<total; i++)
                C.values[i] = A.values[i]-B.values[i];
        }
        else {
            for(int y=0; y<height; y++)
                for(int x=0; x<width; x++)
                    C.at(y,x) = A.at(y,x)-B.at(y,x);
        }
    }
    
    Matrix operator-(const Matrix<T>& B) const {
        Matrix C(B.getHeight(), B.getWidth());
        subtract(*this, B, C);
        return C;
    }
    
    Matrix operator-=(const Matrix<T>& B) {
        if(getWidth() != B.getWidth() || getHeight() != B.getHeight())
            throw MatrixMismatchError();
        
        uint h = getHeight(); uint w = getWidth();
        if(rmnt() == B.rmnt()) {
            for(int i=0; i<total; i++)
                values[i] -= B.values[i];
        }
        else {
            for(int y=0; y<h; y++)
                for(int x=0; x<w; x++)
                    at(y,x) -= B.at(y,x);
        }
        return *this;
    }
    
    void friend hadamard(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
        if(A.getWidth() != B.getWidth() || A.getHeight() != B.getHeight()
           || A.getWidth() != C.getWidth() || A.getHeight() != C.getHeight())
            throw MatrixMismatchError();
        
        //uint total = A.total;
        uint width = A.getWidth(), height = A.getHeight();
        
        
        for(int y=0; y<height; y++)
            for(int x=0; x<width; x++)
                C.at(y,x) = A.at(y,x)*B.at(y,x);
        
        /*
        if(A.rm == B.rm && A.transposed == B.transposed) {
            C.rm = A.rm;
            for(int i=0; i<total; i++)
                C.values[i] = A.values[i]*B.values[i];
        }
        else if(A.rm && B.rm) {
            const Matrix<T>& Tr = A.transposed ? A : B;
            const Matrix<T>& Nt = A.transposed ? B : A;
            
            T *array = new T[total];
            for(int i=0; i<total; i++) {
                uint index = (i/height)+(i%height)*width;
                array[index] = Tr.values[i];
            }
            
            for(int i=0; i<total; i++)
                C.values[i] = Nt.values[i]*array[i];
            
        }
        else if(A.rm != B.rm && A.transposed == B.transposed) {
            const Matrix<T>& Cm = A.rm ? B : A;
            const Matrix<T>& Rm = A.rm ? A : B;
            
            T *array = new T[total];
            for(int i=0; i<total; i++) {
                uint index = (i/height)+(i%height)*width;
                array[index] = Cm.values[i];
            }
            
            for(int i=0; i<total; i++)
                C.values[i] = Rm.values[i]*array[i];
        }
        else {
            for(int y=0; y<height; y++)
                for(int x=0; x<width; x++)
                    C.at(y,x) = A.at(y,x)*B.at(y,x);
        }*/
    }
    
    Matrix operator^(const Matrix<T>& B) {
        Matrix C(getHeight(), getWidth());
        hadamard(*this, B, C);
        return C;
    }
    
    // Vectorized function application
    void apply(void (*func)(T&)) {
        for(int i=0; i<total; i++) func(values[i]);
    }
    
    Matrix apply(void (*func)(T&), bool) {
        Matrix C;
        C.width = getWidth(); C.height = getHeight();
        C.total = total;
        C.rm = true; C.transposed = false;
        T *array = new T[total];
        for(int i=0; i<total; i++) {
            array[i] = values[i];
            func(array[i]);
        }
        C.values = array;
        return C;
    }
    
    T normSqr() const {
        T nsqr = 0;
        for (int i=0; i<total; i++)
            nsqr += (values[i]*values[i]);
        return nsqr;
    }
    
    T sum() const {
        T sum = 0;
        for (int i=0; i<total; i++)
            sum += values[i];
        return sum;
    }
    
    void resize(uint height, uint width) {
        if (width*height!=total) throw MatrixMismatchError();
        
        this->width = width;
        this->height = height;
    }
    
    // Utility functions
    friend std::string mathematicaForm(const Matrix<T>& M, bool semicolon=true) {
        std::stringstream stream;
        stream << "{";
        for(int y=0; y<M.getHeight(); y++) {
            stream << "{";
            for(int x=0; x<M.getWidth(); x++) {
                T num = M.at(y,x);
                if(fabs(num) > 1000000) stream << abs(num)/num * 1000000; // Prevent sci note
                else if(fabs(num) > 0.0001) stream << num; // Prevent sci note
                else stream << 0;
                if(x != M.getWidth()-1)
                    stream << ",";
            }
            stream << "}";
            if(y != M.getHeight()-1)
                stream << ",";
        }
        stream << "}";
        if(semicolon) stream << ";";
        std::string str;
        stream >> str;
        return str;
    }
    
    static Matrix randMatrix(uint rows, uint cols) {
        Matrix M;
        uint total = rows*cols;
        float *values = new float[total];
        if(typeid(T) == typeid(int))
            for(int i=0; i<total; i++) values[i] = intDist(generator);
        else
            for(int i=0; i<total; i++) values[i] = floatDist(generator);
        
        M.width = cols; M.height = rows; M.total = total;
        M.values = values;
        
        return M;
    }
    
    // Accessors
    T& at(uint row, uint col) {
        uint a = row, b = col, c = height;
        if(rmnt()) { a = col; b = row; }
        if(rm) c = width;
        uint index = a + b*c;
        if(index < total) return values[index];
        throw MatrixOutOfBounds();
    }
    
    T at(uint row, uint col) const {
        uint a = row, b = col, c = height;
        if(rmnt()) { a = col; b = row; }
        if(rm) c = width;
        uint index = a + b*c;
        if(index < total) return values[index];
        throw MatrixOutOfBounds();
    }
    
    uint getWidth() const { return transposed ? height : width; }
    uint getHeight() const { return transposed ? width : height; }
    uint getTotal() const { return total; }
    bool isTrans() { return true; }
    bool isRM() const { return rm; }
    
    // Mutators
    void clear() { for(int i=0; i<total; i++) values[i] = 0; }
    void trans() { transposed = !transposed; }
    void neg() { for(int i=0; i<total; i++) values[i] = -values[i]; }
    
private:
    T* __restrict__ values __attribute__(( aligned(sizeof(T)) ));
    
    uint width, height, total;
    bool rm;         // Is this a row major matrix?
    bool transposed; // Is this matrix transposed or not
    
    // We assume that these functions will not try to go out of bounds
    inline T rmAt(uint row, uint col) const { return transposed ? values[col*width+row] : values[row*width+col]; }
    inline T cmAt(uint row, uint col) const { return transposed ? values[row*height+col] : values[col*height+row]; }
    inline T& rmAt(uint row, uint col)      { return transposed ? values[col*width+row] : values[row*width+col]; }
    inline T& cmAt(uint row, uint col)      { return transposed ? values[row*height+col] : values[col*height+row]; }
    
    inline T get(uint row, uint col) const {
        uint choice = (rm << 1) + transposed;
        switch(choice) {
            case 0: // !rm && !transposed
                return values[col*height+row];
                break;
            case 1: // !rm && transposed
                return values[row*height+col];
                break;
            case 2: // rm && !transposed
                return values[row*width+col];
                break;
            case 3: // rm && transposed
                return values[col*width+row];
                break;
            default:
                return T(0);
                break;
        }
    }
    
    inline T& get(uint row, uint col) {
        uint choice = (rm << 1) + transposed;
        switch(choice) {
            case 0: // !rm && !transposed
                return values[col*height+row];
                break;
            case 1: // !rm && transposed
                return values[row*height+col];
                break;
            case 2: // rm && !transposed
                return values[row*width+col];
                break;
            case 3: // rm && transposed
                return values[col*width+row];
                break;
        }
    }
    
    inline bool rmnt() const {return rm!=transposed;} // Row major == not transposed
};

// Function declarations
template<typename T>
inline void writeToStream(const Matrix<T>& M, std::ostream& out) {
    out << "{";
    for(int y=0; y<M.getHeight(); y++) {
        out << "{";
        for(int x=0; x<M.getWidth(); x++) {
            out << M.at(y,x);
            if(x != M.getWidth()-1) out << ",";
        }
        out << "}";
        if(y != M.getHeight()-1) out << ",";
    }
    out << "}";
    // No endline
}

template<typename T>
inline void saveFormat(const Matrix<T>& M, std::ostream& out) {
    out << M.getWidth() << " " << M.getHeight() << " ";
    for(int x=0; x<M.getWidth(); x++)
        for(int y=0; y<M.getHeight(); y++)
            out << M.at(y,x) << " ";
}

template<typename T>
inline std::pair<uint,uint> getMaxIndex(const Matrix<T>& M) {
    std::pair<uint, uint> index;
    T max = M.at(0,0);
    for(int y=0; y<M.getHeight(); y++)
        for(int x=0; x<M.getWidth(); x++)
            if(M.at(y,x) > max) {
                max = M.at(y,x);
                index = std::pair<uint,uint>(y,x);
            }
    return index;
}

template<typename T>
inline Matrix<T> loadFormat(std::istream& in) {
    uint width, height;
    in >> width >> height;
    Matrix<T> M(width, height);
    for(int x=0; x<M.getWidth(); x++)
        for(int y=0; y<M.getHeight(); y++)
            in >> M.at(y,x);
    return M;
}

#endif /* defined(__Empack__Matrix__) */