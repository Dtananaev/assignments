import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class MyConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    #conv 1
    self.params['W1'] = weight_scale * np.random.randn(num_filters,input_dim[0],filter_size,filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['gamma1'] = np.ones(num_filters)
    self.params['beta1'] = np.zeros(num_filters)
    #conv 2
    self.params['W2'] = weight_scale * np.random.randn(num_filters,num_filters,filter_size,filter_size)
    self.params['b2'] = np.zeros(num_filters)
    self.params['gamma2'] = np.ones(num_filters)
    self.params['beta2'] = np.zeros(num_filters)
    #conv 3
    self.params['W3'] = weight_scale * np.random.randn(num_filters,num_filters,filter_size,filter_size)
    self.params['b3'] = np.zeros(num_filters)
    self.params['gamma3'] = np.ones(num_filters)
    self.params['beta3'] = np.zeros(num_filters)
    #pool
    size_pool_1=(input_dim[1]-2)/2 +1
    size_pool_2=(input_dim[2]-2)/2 +1    
    input_conv=num_filters*size_pool_1*size_pool_2
    self.params['W4'] = weight_scale * np.random.randn(input_conv, num_classes)
    self.params['b4'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    self.bn_params = {'mode': 'train'}
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #conv1
    out_1, cache_1 =conv_forward_strides(X, W1, b1, conv_param)
    out_b1,cache_b1=spatial_batchnorm_forward(out_1, self.params['gamma1'], self.params['beta1'], self.bn_params)
    out_r1,cache_r1=relu_forward(out_b1)
    #conv2
    out_2, cache_2 =conv_forward_strides(out_r1, W2, b2, conv_param)
    out_b2,cache_b2=spatial_batchnorm_forward(out_2, self.params['gamma2'], self.params['beta2'], self.bn_params)
    out_r2,cache_r2=relu_forward(out_b2)
    #conv3
    out_3, cache_3 =conv_forward_strides(out_r2, W3, b3, conv_param)
    out_b3,cache_b3=spatial_batchnorm_forward(out_3, self.params['gamma3'], self.params['beta3'], self.bn_params)
    out_r3,cache_r3=relu_forward(out_b3)
    #pool
    out_4, cache_4  =max_pool_forward_fast(out_r3, pool_param)
    #affine1
    scores, cache_5 = affine_forward(out_4, W4, b4)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout1    = softmax_loss(scores, y)
    sum_weights=np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3)
    loss+=0.5*self.reg*sum_weights
    #affine2
    dout2,grads["W4"],grads["b4"]=affine_backward(dout1, cache_5)
    #pool
    dout4=max_pool_backward_fast(dout2, cache_4)
    #conv3
    dout5=relu_backward(dout4,cache_r3)
    doutB5,grads['gamma3'],grads['beta3']=spatial_batchnorm_backward(dout5, cache_b3)
    dout6,grads["W3"],grads["b3"]=conv_backward_strides(doutB5, cache_3)
    #conv2
    dout7=relu_backward(dout6,cache_r2)
    doutB7,grads['gamma2'],grads['beta2']=spatial_batchnorm_backward(dout7, cache_b2)
    dout8,grads["W2"],grads["b2"]=conv_backward_strides(doutB7, cache_2)
    #conv1
    dout9=relu_backward(dout8,cache_r1)
    doutB9,grads['gamma1'],grads['beta1']=spatial_batchnorm_backward(dout9, cache_b1)    
    dout10,grads["W1"],grads["b1"]=conv_backward_strides(doutB9, cache_1)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
