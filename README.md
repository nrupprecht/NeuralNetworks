# NeuralNetworks

This is the code for our neural network project.

The make file should work fine.

You can ignore everything in the file "Files." 

MNISTData contains raw data files of the MNIST data, use the FileUnpack class in MNISTUnpack.h to access the data and put it into a reasonable format.

MNISTNet is a program that sets up a network to learn the MNIST dataset. It acheives about 95% accuracy on the test set within 5 iteration.

EasyBMP is a useful little program someone (Paul Macklin) wrote to handle BMP files. I use it all the time, its great. Don't modify it though. That would be unneccesary.

The Matrix code is mostly just a wrapper for blas that allows us to port around matrices and their associated data a lot easier.
