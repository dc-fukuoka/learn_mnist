learn_mnist - neural network learning with MNIST data by python.
===
requirements: numpy  
  
Basically, the code is almost the same as https://github.com/mnielsen/neural-networks-and-deep-learning , though there are several modifications.  
Some modifications(except for my notes/comments):  
 * use the original MNIST data  
 * use python3  
 * replace np.dot() with np.outer() for a calculation of nabla_w in backward propagation, based on the algorithm  
  
The purpose of this program is for study of python and to understand how neural network/deeplearning works.  

how to test:  
====
~~~
$ cd data/
$ wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
$ wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
$ wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
$ wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
$ cd ../
$ vi main.py # adjust the parameters such as sizes, mini_batch_size
$ ./main.py
sizes:           (784, 30, 10)
epochs:          30
mini_batch_size: 100
eta:             3.000000
Epoch  0:  4766 / 10000: 47.66 %
Epoch  1:  4661 / 10000: 46.61 %
Epoch  2:  5481 / 10000: 54.81 %
Epoch  3:  5842 / 10000: 58.42 %
Epoch  4:  6540 / 10000: 65.40 %
Epoch  5:  6352 / 10000: 63.52 %
Epoch  6:  6159 / 10000: 61.59 %
Epoch  7:  6435 / 10000: 64.35 %
Epoch  8:  6494 / 10000: 64.94 %
Epoch  9:  6888 / 10000: 68.88 %
Epoch 10:  6899 / 10000: 68.99 %
Epoch 11:  6934 / 10000: 69.34 %
Epoch 12:  6738 / 10000: 67.38 %
Epoch 13:  6432 / 10000: 64.32 %
Epoch 14:  6570 / 10000: 65.70 %
Epoch 15:  6317 / 10000: 63.17 %
Epoch 16:  7139 / 10000: 71.39 %
Epoch 17:  6946 / 10000: 69.46 %
Epoch 18:  6613 / 10000: 66.13 %
Epoch 19:  6833 / 10000: 68.33 %
Epoch 20:  6784 / 10000: 67.84 %
Epoch 21:  6933 / 10000: 69.33 %
Epoch 22:  7389 / 10000: 73.89 %
Epoch 23:  7219 / 10000: 72.19 %
Epoch 24:  7062 / 10000: 70.62 %
Epoch 25:  6894 / 10000: 68.94 %
Epoch 26:  7070 / 10000: 70.70 %
Epoch 27:  7209 / 10000: 72.09 %
Epoch 28:  7366 / 10000: 73.66 %
Epoch 29:  7541 / 10000: 75.41 %
time[s]: 280.774842
~~~
  
Still accuracy rate is not so high, need to check the code.
