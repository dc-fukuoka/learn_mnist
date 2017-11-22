import random
import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prim(x):
    # analyticial derivtative of sigmoid(x)
    return (1.0-sigmoid(x))*sigmoid(x)

# MSE
# 1/2*SIGMA_i(a^L_i - y_i)^2
# we want to minimize this, so only need its gradient
def cost(y, a):
    size = len(y)
    mse  = 0.0
    for i in range(size):
        mse += 0.5*(a[i] - y[i])**2
    return mse

# derivative of MSE, just a^L_i - y_i
def nabla_cost(a, y):
    return (a - y)

class Network():
    def __init__(self, sizes):
        self.sizes   = sizes
        self.nlayers = len(sizes)
        self.biases  = [np.random.randn(x).astype(np.float32)    for x    in sizes[1:]]
        self.weights = [np.random.randn(y, x).astype(np.float32) for x, y in zip(sizes[:-1], sizes[1:])]

        # ex: for a case sizes = (4, 3, 8, 2)
        # nlayers = 4
        # zip((4,3,8), (3,8,2)) -> (4,3),(3,8),(8,2)
        # W_0(3,4) W_1(8,3) W_2(2,8)
        # b_0(3)   b_1(8)   b_2(2)
        #
        # a_1(3) = sigma(W_0(3,4)*a_0(4)+b_0(3))
        # a_2(8) = sigma(W_1(8,3)*a_1(3)+b_1(8))
        # a_3(2) = sigma(W_2(2,8)*a_2(8)+b_2(2))

    # feed forward
    # return sigma(W*a+b)
    def fforward(self, a):
        # calculate and update a = sigma(W*a+b) for each layer
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    # backward propagation, calculate dC/dw and dC/db to minimize C(w, b)
    # input:  x: images y: labels
    # return: (nabla_b, nabla_w)
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # ex: for a case sizes = (4, 3, 8, 2)
        # nabla_b = ((0 x3), (0 x8), (0 x2))
        # nabla_w = ((0 x4x3), (0 x3x8), (0 x8x2))

        # feed forward
        activation  = x
        activations = [x] # to store sigma(z)
        zs          = []  # to store z = w*a + b
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b # z = w*a + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backword propagation
        # delta^L = dC/da*sigma'(z^L)
        delta = nabla_cost(activations[-1], y)*sigmoid_prim(zs[-1]) # here * is the Hadmard product: a(i) * b(i) = (a(1)*b(1), a(2)*b(2),...), numpy ndarray product does this
        # delta^l = (w^(l+1)^T*delta^(l+1))*sigma'(z^l)
        # dC/db^l_j = delta^l_j
        nabla_b[-1] = delta
        # dC/dw^L_jk = a^(L-1)_k*delta^L_j
        nabla_w[-1] = np.outer(delta, activations[-2]) # activations[-2] = activations[L-1], note: this is the outer product, see the algorithm

        # -2, -3, -4,... -nlayers
        # layer by layer
        for l in range(2, self.nlayers):
            delta       = np.dot(self.weights[-l+1].transpose(), delta)*sigmoid_prim(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.outer(delta, activations[-l-1]) # note: this is the outer product, see the algorithm
        
        return (nabla_b, nabla_w)

    def eval(self, test_data):
        test_results = [(np.argmax(self.fforward(x)), np.argmax(y)) for x, y in test_data]
        # return # of matches with the labels
        return sum(iter(int(x == y) for x, y in test_results))

    # update biases and weights int the mini batch at once
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch: # x: images y: labels in the mini batch
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # sum of gradients in the mini batch
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.biases  = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases,  nabla_b)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]

    # stochastic gradient descent
    def SGD(self, train_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data is not None:
            ntests = len(test_data)
        ndata = len(train_data)
        for e in range(epochs):
            random.shuffle(train_data)
            mini_batches = [train_data[k:k+mini_batch_size] for k in range(0, ndata, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data is not None:
                nmatches = self.eval(test_data)
                print("Epoch %2d: %5d / %5d: %4.2lf %%" % (e, nmatches, ntests, nmatches/ntests*100.0))
            else:
                print("Epoch %d" % (e))
