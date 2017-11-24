learn_mnist - neural network learning with MNIST data by python.
===
requirements: numpy  
  
Basically, the code is almost the same as https://github.com/mnielsen/neural-networks-and-deep-learning , though there are several modifications.  
Some modifications are below(except for my notes/comments):  
 * use the original MNIST data  
 * use python3  
 * use n dimesion array insead of (n, 1) dimension array for image data and Network.biases  
 * replace np.dot() with np.outer() for a calculation of nabla_w in backward propagation, based on the algorithm  
  
The purpose of this program is for study of python and to understand how neural network/deeplearning works.  

how to test:
====
CPU was Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz 20 cores.
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
mini_batch_size: 10
eta:             3.000000
Epoch  0:  8297 / 10000: 82.97 %
Epoch  1:  8457 / 10000: 84.57 %
Epoch  2:  8451 / 10000: 84.51 %
Epoch  3:  8497 / 10000: 84.97 %
Epoch  4:  8542 / 10000: 85.42 %
Epoch  5:  9408 / 10000: 94.08 %
Epoch  6:  9422 / 10000: 94.22 %
Epoch  7:  9447 / 10000: 94.47 %
Epoch  8:  9418 / 10000: 94.18 %
Epoch  9:  9467 / 10000: 94.67 %
Epoch 10:  9445 / 10000: 94.45 %
Epoch 11:  9484 / 10000: 94.84 %
Epoch 12:  9481 / 10000: 94.81 %
Epoch 13:  9498 / 10000: 94.98 %
Epoch 14:  9468 / 10000: 94.68 %
Epoch 15:  9491 / 10000: 94.91 %
Epoch 16:  9510 / 10000: 95.10 %
Epoch 17:  9509 / 10000: 95.09 %
Epoch 18:  9504 / 10000: 95.04 %
Epoch 19:  9494 / 10000: 94.94 %
Epoch 20:  9512 / 10000: 95.12 %
Epoch 21:  9482 / 10000: 94.82 %
Epoch 22:  9535 / 10000: 95.35 %
Epoch 23:  9516 / 10000: 95.16 %
Epoch 24:  9502 / 10000: 95.02 %
Epoch 25:  9523 / 10000: 95.23 %
Epoch 26:  9537 / 10000: 95.37 %
Epoch 27:  9521 / 10000: 95.21 %
Epoch 28:  9522 / 10000: 95.22 %
Epoch 29:  9507 / 10000: 95.07 %
time[s]: 301.463009
~~~
