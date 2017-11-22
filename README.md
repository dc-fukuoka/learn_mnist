learn_mnist - neural network learning with MNIST data by python. in progress.
===
requirements: numpy  
  
Basically, the code is almost the same as https://github.com/mnielsen/neural-networks-and-deep-learning , though there are several modifications.  
Some modifications are below(except for my notes/comments):  
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
Epoch  0:  5691 / 10000: 56.91 %
Epoch  1:  6863 / 10000: 68.63 %
Epoch  2:  7142 / 10000: 71.42 %
Epoch  3:  7244 / 10000: 72.44 %
Epoch  4:  7953 / 10000: 79.53 %
Epoch  5:  8208 / 10000: 82.08 %
Epoch  6:  8272 / 10000: 82.72 %
Epoch  7:  8310 / 10000: 83.10 %
Epoch  8:  8343 / 10000: 83.43 %
Epoch  9:  8369 / 10000: 83.69 %
Epoch 10:  8506 / 10000: 85.06 %
Epoch 11:  9095 / 10000: 90.95 %
Epoch 12:  9142 / 10000: 91.42 %
Epoch 13:  9167 / 10000: 91.67 %
Epoch 14:  9192 / 10000: 91.92 %
Epoch 15:  9229 / 10000: 92.29 %
Epoch 16:  9238 / 10000: 92.38 %
Epoch 17:  9244 / 10000: 92.44 %
Epoch 18:  9272 / 10000: 92.72 %
Epoch 19:  9278 / 10000: 92.78 %
Epoch 20:  9289 / 10000: 92.89 %
Epoch 21:  9298 / 10000: 92.98 %
Epoch 22:  9307 / 10000: 93.07 %
Epoch 23:  9309 / 10000: 93.09 %
Epoch 24:  9319 / 10000: 93.19 %
Epoch 25:  9327 / 10000: 93.27 %
Epoch 26:  9326 / 10000: 93.26 %
Epoch 27:  9338 / 10000: 93.38 %
Epoch 28:  9326 / 10000: 93.26 %
Epoch 29:  9343 / 10000: 93.43 %
time[s]: 276.157600
~~~
  
Still accuracy rate is not so high, need to check the code.
