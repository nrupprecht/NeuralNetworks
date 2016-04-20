CC = icpc
MPICC = mpicxx
OPT = -O3 -g
# -xHost is slow, fo is -fast since it includes xHost
CFLAGS = -std=c++14 $(OPT)
MKLROOT = /afs/crc.nd.edu/x86_64_linux/intel/15.0/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

targets = MNISTNet CIFARNet AutoEncodeMNIST
base = Network.o Neuron.o Tensor.o
all:	$(targets)

# Executables
MNISTNet : MNISTNet.o $(base) MNISTUnpack.o EasyBMP.o
	$(MPICC) -o $@ $^ $(LDLIBS)

CIFARNet: CIFARNet.o $(base) CIFARUnpack.o
	$(MPICC) -o $@ $^ $(LDLIBS)

AutoEncodeMNIST: AutoEncodeMNIST.o $(base) MNISTUnpack.o EasyBMP.o
	$(MPICC) -o $@ $^ $(LDLIBS)

# Object files
EasyBMP.o : EasyBMP/EasyBMP.cpp
	$(CC) -c $(CFLAGS) $<

Network.o : Network.cpp Neuron.o
	$(MPICC) -c $(CFLAGS) $<

%.o : %.cpp
	$(MPICC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) *.o *.stdout
