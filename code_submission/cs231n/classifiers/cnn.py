from builtins import object
import numpy as np

#from ..layers import *
from ..fast_layers import *
from ..layer_utils import *
from cs231n.layers import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
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
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #pirobashi weria, rom w1/w2/w3 and b1/b2/b3-is inicializacia
        #gvinda jer amovigot parametrebi
        #jer avigem dimensionebs
        a,b,c = input_dim
        
        fz = filter_size
        norm_b,norm_c = b//2, c//2
        dimension_two = norm_b* norm_c
        full_dimension = num_filters* dimension_two

        w1_random=np.random.randn(num_filters,a,fz,fz)
        w2_random=np.random.randn(full_dimension,hidden_dim)
        w3_random=np.random.randn(hidden_dim,num_classes)
        b1_zer= num_filters
        b2_zer= hidden_dim
        b3_zer= num_classes

        #saboloo weights and biases 
        W1= weight_scale* w1_random
        W2= weight_scale*w2_random
        W3=weight_scale*w3_random
        self.params['W1']=W1
        self.params['W2']=W2
        self.params['W3']=W3

        b1=np.zeros( b1_zer)
        b2=np.zeros(b2_zer)
        b3=np.zeros(b3_zer)
        self.params['b1']=b1
        self.params['b2']=b2
        self.params['b3']=b3

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #radgan layer.py-shi ukve miweria conv_relu_pool_forward/conv_relu_forward
        #da bolo layeristvis, affine_forward, ubralod magat gamoviyeneb chems shemtxvevashi

        #jer forward pass-s davwer conv_relu_pool_forwardit
        out1 =conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
        param1_out, mem1_out = out1
        #forward pass affine-relu-ti
        out2 = affine_relu_forward(param1_out,W2,b2)
        param2_out, mem2_out = out2
        #forward pass bolo gareta layeria da amitomaa marto affine
        out3 = affine_forward(param2_out, W3,b3)
        cul_track, mem_meore_affine = out3
        scores = cul_track

        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        grads={}
        sr =self.reg
        sr_W1 = sr*W1
        sr_W2 = sr*W2
        sr_W3= sr*W3

        kva3 =self.reg * W3
        kva2=self.reg * W2
        kva1=self.reg * W1

        comp = sr/2.0
        wonebi = [W1,W2,W3]
        gasworebuli_loss = comp* sum(np.sum(won*won) for won in wonebi)


        #jer mwirdeba softmax_loss-idan gamovidzaxo da davtvalo chemi loss da monacemebi
        soft_loss, cul_two_track = softmax_loss(scores,y)
        #loss maqvs, magradm wirdeba damushaveba 
        full_loss = gasworebuli_loss + soft_loss


        #backdrop of output pena
        out_func1= affine_backward(cul_two_track,mem_meore_affine)
        out1_1,out1_2,out1_3=out_func1
        af_back_x =out1_1
        af_back_w =out1_2
        af_back_b=out1_3
        wona_1 =  af_back_w + kva3
        bias_1 = af_back_b
        grads['W3']=wona_1
        grads['b3']=bias_1
        #hidden pena
        out_func2=affine_relu_backward(af_back_x ,mem2_out)
        out2_1,out2_2,out2_3=out_func2
        af_relu_back_x =out2_1
        af_relu_back_w =out2_2
        af_relu_back_b=out2_3
        wona_2 = af_relu_back_w + kva2
        bias_2 = af_relu_back_b
        grads['W2']=wona_2
        grads['b2']=bias_2
        #input pena 
        out_func3 = conv_relu_pool_backward(out2_1, mem1_out)
        out3_1,out3_2,out3_3=out_func3
        af_rp_back_x =out3_1
        af_rp_back_w =out3_2
        af_rp_back_b=out3_3
        wona_3 =af_rp_back_w + kva1
        bias_3 = af_rp_back_b
        grads['W1']=wona_3
        grads['b1']=bias_3

        loss = full_loss


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
