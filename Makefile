CC = icpc
OPT = -fast -xHost -restrict -no-prec-div
CFLAGS = -std=c++14 $(OPT)
MKLROOT = /afs/crc.nd.edu/x86_64_linux/intel/15.0/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

targets = MNISTNet
objects = Network.o Matrix.o MNISTUnpack.o EasyBMP.o MNISTNet.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

MNISTNet : MNISTNet.o Network.o Matrix.o MNISTUnpack.o EasyBMP.o
	$(CC) -o $@ $^ $(LDLIBS)

EasyBMP.o : EasyBMP/EasyBMP.cpp
	$(CC) -c $(CFLAGS) $<

%.o : %.cpp
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects) *.stdout
