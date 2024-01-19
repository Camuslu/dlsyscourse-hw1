"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np
from python.needle.ops import *
import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, 'rb') as f:
        x_raw = np.frombuffer(f.read(), 'B', offset=16) # the magic number, num_image, num_row, num_col are the offset (4 * 4 bytes)
    x_reshaped = x_raw.reshape(-1, 784).astype('float32')
    x_reshaped = x_reshaped/255

    with gzip.open(label_filename, 'rb') as f:
        y_raw = np.frombuffer(f.read(), 'B', offset=8) # the magic number, num_image, num_row, num_col are the offset (4 * 4 bytes)
    y_reshaped = y_raw.reshape(-1).astype('uint8')
    return x_reshaped, y_reshaped
    ### END YOUR CODE

def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    batch_size = Z.shape[0]
    Z_exp = exp(Z) # [batch_size, k_class]
    Z_sum_exp = summation(Z_exp, axes=(1,)) # [batch_size]
    Z_log_sum_exp = log(Z_sum_exp) # [batch_size]

    Z_y = EWiseMul()(Z, y_one_hot) # [batch_size, k_class]
    Z_y_sum = summation(Z_y, axes=(1,)) # [batch_size]
    loss = Z_log_sum_exp - Z_y_sum
    return summation(loss) / batch_size # for average loss in batch
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    ### BEGIN YOUR SOLUTION
    classes = len(np.unique(y))
    for step in range(X.shape[0]//batch):
        X_batch = X[step * batch: (step+1) * batch, :] # [b_size, x_dim]
        y_batch = y[step * batch: (step+1) * batch] # [b_size]
        one_hot_label = np.eye(classes)[y_batch] # [b_size, class]
        X_batch_tensor = Tensor(X_batch) #[b_size, x_dim]
        y_batch_tensor = Tensor(one_hot_label) #[b_size, class]
        layer_1 = relu(matmul(X_batch_tensor, W1)) #[b_size, hiddem_dim]
        layer_2 = matmul(layer_1, W2) #[b_size, class]
        loss = softmax_loss(layer_2, y_batch_tensor) # already averaged over batch_size
        loss.backward()
        W1_grad = W1.grad.realize_cached_data()
        W1 = Tensor(W1.realize_cached_data() - W1_grad * lr)
        W2_grad = W2.grad.realize_cached_data()
        W2 = Tensor(W2.realize_cached_data() - W2_grad * lr)          
    return (W1, W2)      
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
