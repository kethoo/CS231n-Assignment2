from builtins import range
from builtins import object
import numpy as np
from numpy.random import gamma

from ..layers import *
from ..layer_utils import *

class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #firstly, lets initialize the weights and biases for each layers
        #for that, i will need to get the full dimensions and i will
        #look at the init section to get the desired variables needed for
        #further work
        prop = []
        prop.append(input_dim)
        prop.extend(hidden_dims) #this is already a list 
        prop.append(num_classes)

        #now that i have correct number, ill iterate and initialize w and b
        to_iterate_over = self.num_layers +1 #get the amount of layers

        for i in range(1, to_iterate_over):
          mv = prop[i] #this will be used in both!!!
          kv = prop[i-1]
          to_mul_bw = np.random.randn(kv,mv)
          #scale this by the weight
          #weights myst be randrom normal distribution
          final_w = weight_scale* to_mul_bw
          #biases must be zero 
          final_b = np.zeros(mv)

          #assign weight and bias to it 
          self.params['W' + str(i)] = final_w
          self.params['b' + str(i)] = final_b

          #case of batch or layer norm
          batch_layer = ["batchnorm","layernorm"]
          

          if self.normalization in batch_layer and i < self.num_layers :
            final_gamma = np.ones(mv) #init to 1
            final_beta = np.zeros(mv) #init to 0
            self.params['gamma' + str(i)] = final_gamma
            self.params['beta' + str(i)] = final_beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #jer only implement forward pass:
        #for every hidden layer affine -> batch norm -> relu ->droput
        #for final layer marto affine
        #give scores  

        #save for forward passing
        x = X
        mem_total = {}

        total_layer = self.num_layers
        for i in range (1, total_layer):
          #remember the weights 
          wgs = self.params['W' + str(i)]
          bs = self.params['b'+str(i)]
          out, aff_mem = affine_forward(x,wgs,bs)

          normal_mem = None
          norm_cases = ["batchnorm","layernorm"]
          if self.normalization == "batchnorm":
            norm_func = batchnorm_forward
          elif self.normalization == "layernorm":
            norm_func = layernorm_forward

          if self.normalization in norm_cases:
            gamma = self.params['gamma' + str(i)]
            beta = self.params['beta' + str(i)]
            out, normal_mem = norm_func(out, gamma, beta,self.bn_params[i-1])
          

          #apply relu
          out, relu_mem = relu_forward(out)

          #case of the dropout 
          dropout_mem = None
          if self.use_dropout:
            out, dropout_mem = dropout_forward(out, self.dropout_param)

          mem_total[i] = (aff_mem, normal_mem,relu_mem,dropout_mem)
          x =out 

        #pass the parameters and implement the outside layer
        #this only does the affine stuff 
        wei = self.params['W' + str(total_layer)]
        bis = self.params['b' +str(total_layer)]
        scores, aff_mem = affine_forward(x,wei,bis)
        mem_total[total_layer]= aff_mem 



        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #now i shall compute softmax loss and gradient of it using a formula 
        #backpropagation of gradients, and keep them in grads! and use L2 (depresia)
        grads = {}

        #now will do softmax 
        loss, depre = softmax_loss(scores,y)
        regul_loss = 0.0
        full_layer = self.num_layers+1
        exceptl_layer = self.num_layers
        for i in range(1, full_layer):
          wei = self.params['W' + str(i)]
          sum_calc = np.sum(wei*wei)
          regul_loss =regul_loss + sum_calc
        
        #after regularisation L2 happend on loss, ill add it to the whole loss
        loss = loss + 0.5*self.reg*regul_loss

        #after this, lets look at the last layer 
        mem_last_layer = mem_total[exceptl_layer]
        dout, dW, db = affine_backward(depre,mem_last_layer)
        #keep track of those gradients 
        grads['W'+str(exceptl_layer)] = dW
        grads['b'+str(exceptl_layer)] =db

        #done with the last layer, now must look at the hidden layers
        #backward some stuff here and iterate over them
        for curr_done in reversed(range(1,exceptl_layer)):
          aff_mem, normal_mem, relu_mem, dropout_mem = mem_total[curr_done]

          #if dropout
          if self.use_dropout:
            dout = dropout_backward(dout, dropout_mem)

          #now relu
          dout = relu_backward(dout, relu_mem)

          #now, as always, lets take a look if normalization is used and what 
          #is happening in this case
          if self.normalization == 'layernorm':
            dout, gam, bet = layernorm_backward(dout, normal_mem)
            grads['gamma' + str(curr_done)] = gam
            grads['beta' + str(curr_done)] = bet 
          
          elif self.normalization == 'batchnorm':
            dout, gam, bet = batchnorm_backward(dout, normal_mem)
            grads['gamma' + str(curr_done)] = gam
            grads['beta' + str(curr_done)] = bet 

          dout, dW, db = affine_backward(dout, aff_mem)
          #adding the gradients 
          grads['W' + str(curr_done)] = dW
          grads['b' + str(curr_done)] = db

        #weight modification as well
        #ES GAMOGRCHA TAVIDAN DA MAGITO DAGIJDA MILIONJER METI ERRORI
        for ind in range(1, full_layer):
          curr_weight = self.params['W' + str(ind)]
          full_additional_wreg = self.reg*curr_weight
          grads['W' +str(ind)] =  grads['W' +str(ind)] + full_additional_wreg



        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
