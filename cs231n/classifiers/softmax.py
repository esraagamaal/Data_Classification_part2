import numpy as np
from random import shuffle

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
  
  
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  for i in range(num_train):

    fun_i = X[i].dot(W)
    fun_i -= np.max(fun_i)

    sum_prob = np.sum(np.exp(fun_i))
    
    
    #calculate all the prob for whole labels
    prob = np.exp(fun_i)/sum_prob
    
    #the psrob of the real label
    loss += -np.log(prob[y[i]])

    for j in range(num_classes):
      #get all dw values for each label j
      #-ve if same label and prob ---> cost decrease
      dW[:, j] += (prob[j] - (j == y[i])) * X[i]

  #avg
  loss = loss/num_train
  dW = dW/num_train
  #regularization
  loss = loss + 0.5 * reg * np.sum(W * W)
  dW = dW + reg*W
    
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # vectorizatin
  num_train = X.shape[0]
  fun = X.dot(W)
  fun -= np.max(fun, axis=1, keepdims=True)
  sum_prob = np.sum(np.exp(fun), axis=1, keepdims=True)
  prob = np.exp(fun)/sum_prob

  loss_vec = -np.log(prob[np.arange(num_train), y])
  loss = np.sum(loss_vec)
  if_label = np.zeros(prob.shape)
  if_label[np.arange(num_train), y] = 1 # =1 only for the class == real label
  dW = np.transpose(X).dot(prob - if_label)
  
  #avg
  loss = loss/num_train
  dW = dW/num_train
  #regularization
  loss = loss + 0.5 * reg * np.sum(W * W)
  dW = dW + reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

