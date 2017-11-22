#!/usr/bin/env python3

from read_mnist import read_mnist
from network import Network
import time as T

def main():
    train_data, validation_data, test_data = read_mnist()

    # the first value in sizes must be 28*28 because 28x28 image
    # the last value in sizes must be 10, 0 - 9
    # another example is sizes = (28*28, 40, 30, 10)
    sizes           = (28*28, 30, 10)
    epochs          = 30
    mini_batch_size = 10
    eta             = 3.0
    print("sizes:           ", end='')
    print(sizes)
    print("epochs:          %d"        % (epochs))
    print("mini_batch_size: %d"        % (mini_batch_size))
    print("eta:             %lf"       % (eta))
    
    t0    = T.time()    
    net   = Network(sizes)
    net.SGD(train_data, epochs, mini_batch_size, eta, test_data)
    t1    = T.time() - t0
    print("time[s]: %lf" % (t1))

if __name__ == "__main__":
    main()
