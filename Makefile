CC = icpc
OPT = -O3 #-fast -xHost -restrict -no-prec-div
CFLAGS = -std=c++14 $(OPT)
MKLROOT = /afs/crc.nd.edu/x86_64_linux/intel/15.0/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

targets = MNISTNet CIFARNet
all:	$(targets)

#.PHONY : default
#default : all

# To clear everything before every make, uncomment these lines
#all : clean $(targets)

MNISTNet : MNISTNet.o Network.o Neuron.o Matrix.o MNISTUnpack.o EasyBMP.o
	$(CC) -o $@ $^ $(LDLIBS)

CIFARNet: CIFARNet.o Network.o Neuron.o Matrix.o CIFARUnpack.o
	$(CC) -o $@ $^ $(LDLIBS)

# Object files
EasyBMP.o : EasyBMP/EasyBMP.cpp
	$(CC) -c $(CFLAGS) $<

Network.o : Network.cpp Neuron.o
	$(CC) -c $(CFLAGS) $<

%.o : %.cpp
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) *.o *.stdout
