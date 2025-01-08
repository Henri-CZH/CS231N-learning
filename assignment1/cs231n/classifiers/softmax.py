from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = np.dot(X[i], W) # size: 1xC
        correct_class_score = scores[y[i]] # size: 1x1
        sumExpScore = np.sum(np.exp(scores)) # size: 1x1
        correct_class_expScore = np.exp(correct_class_score) # size: 1x1
        divScore = correct_class_expScore / sumExpScore # size: 1x1
        margin = -np.log(divScore) # size: 1x1
        loss += margin # size: 1x1
        for j in range(num_classes):
            classExpScore = np.exp(scores[j]) # size: 1x1
            remainingClassExpScore = sumExpScore - correct_class_expScore # size: 1x1
            if y[i] == j:
              dW[:, y[i]] += -(1 / divScore) * (correct_class_expScore * remainingClassExpScore / pow(sumExpScore, 2)) * X[i] # size: 1x1
            else:
              dW[:, j] += -(1 / divScore) * (-correct_class_expScore * classExpScore / pow(sumExpScore, 2)) * X[i] # size: 1x1
                
    
    loss = loss/num_train + reg * np.sum(W * W)

    dW = dW/num_train + np.multiply(2 * reg, W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = np.dot(X, W) # size: NxC
    correct_class_score = scores[np.arange(num_train), y].reshape((num_train, 1)) # size: Nx1
    classExpScore = np.exp(scores) # size: NxC
    sumExpScore = np.sum(classExpScore, axis = 1).reshape((num_train, 1)) # size: Nx1
    correct_class_expScore = np.exp(correct_class_score) # size: Nx1
    posteriorScore = correct_class_expScore / sumExpScore # size: NxC
    margin = -np.log(posteriorScore) # size: NxC
    loss = np.sum(margin) / num_train + reg * np.sum(W * W) # size: 1x1

    remainingClassExpScore = sumExpScore - correct_class_expScore # size: Nx1
    tmpExpScore = np.multiply(-classExpScore, correct_class_expScore) # size: NxC
    tmpExpScore[np.arange(num_train), y] = np.multiply(correct_class_expScore, remainingClassExpScore).reshape((num_train, )) # size: NxC
    dW = np.dot(X.T, (-1 / posteriorScore) * (tmpExpScore / np.square(sumExpScore))) # size: DxC

    dW = dW / num_train + np.multiply(2 * reg, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
