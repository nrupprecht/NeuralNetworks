# NeuralNetworks

This is the code for our neural network project.

The make file should work fine.

You can ignore everything in the file "Files." 

MNISTData contains raw data files of the MNIST data, use the FileUnpack class in MNISTUnpack.h to access the data and put it into a reasonable format.
CIFARData contains raw data files of the CIFAR data, use CIFARUnpack to access and format that data.

MNISTNet is a program that sets up a network to learn the MNIST dataset. It can acheive about 95% accuracy on the test set within 5 iteration if you use a network with 784 * 50 * 10 neurons. CIFARNet is a program for classifying the CIFAR dataset.

EasyBMP is a useful little program someone (Paul Macklin) wrote to handle BMP files. I use it all the time, its great. Don't modify it though. That would be unnecesary.

The Matrix code is mostly just a wrapper for blas that allows us to port around matrices and their associated data a lot easier.
