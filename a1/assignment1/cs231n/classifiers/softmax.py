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
  scores=np.dot(X,W)
  max_score=scores.max(1)
  scores=scores - np.tile(max_score,[scores.shape[1],1]).T
  denominator= np.sum(np.exp(scores),axis=1)
  softmax=np.exp( scores[range(scores.shape[0]),y])/denominator
  loss=np.sum(-np.log(softmax))


  for i in range(X.shape[0]):
    for j in range(dW.shape[1]):
      if(j==y[i]):
        dW[:,j]+=X[i,:]*(softmax[i]-1)
      else:
        dW[:,j]+=X[i,:]*np.exp(scores[i,j])/denominator[i]

  loss=loss/X.shape[0]  
  loss+=0.5*reg*np.sum(W*W)
  dW=dW/X.shape[0] 
  dW+=0.5*reg*W


  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  scores=np.dot(X,W)
  max_score=scores.max(1)
  scores=scores - np.tile(max_score,[scores.shape[1],1]).T
  denominator= np.sum(np.exp(scores),axis=1)
  softmax=np.exp( scores[range(scores.shape[0]),y])/denominator
  loss=np.sum(-np.log(softmax))

  temp=np.exp(scores)/ np.tile(denominator,[scores.shape[1],1]).T  
  temp[range(temp.shape[0]),y]-=1;
  dW=np.dot(X.T,temp)
  loss=loss/X.shape[0]
  dW=dW/X.shape[0]
  loss+=0.5*reg*np.sum(W*W)
  dW+=0.5*reg*W
  return loss, dW

