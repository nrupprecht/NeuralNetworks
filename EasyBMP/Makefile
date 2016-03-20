CC = icpc
OPT = -fast -xHost -restrict -no-prec-div
CFLAGS = -std=c++14 $(OPT)

targets = EasyBMP
objects = EasyBMP.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

EasyBMP : EasyBMP.o
	$(CC) -o $@ $^

%.o : %.cpp
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean: rm -f $(targets) $(objects) *.stdout