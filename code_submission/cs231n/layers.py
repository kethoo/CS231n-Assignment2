from builtins import range
import numpy as np
from numpy._core.multiarray import MAY_SHARE_BOUNDS


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    # the whole point of forward affine is to  implement out = xw + b
    to_resize = x.shape[0] 
    # flatten it
    resized_x = x.reshape(to_resize,-1)
    #count the dot product of correct dimension x and w
    weighted_x = np.dot(resized_x,w) 
    #add the bias 
    out = weighted_x + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # i need gradient w.r.t input, gradient w.r.t weights and gradient w.r.t bias
    # and they are dout* W.T , X.T* dout and sum of dout
    #resizing x to 2d shape, flatten it
    to_resize = x.shape[0]

    resized_x = x.reshape(to_resize,-1)

    db = dout.sum(axis =0 ) #sum all the dout
    
    tranp_x = resized_x.T
    dw = tranp_x.dot(dout) 
    #matching original input shape dx
    tranp_weights = w.T
    #backward pass implementation
    ty_war = dout.dot(tranp_weights)
    orig_shape = x.shape
    dx = ty_war.reshape(orig_shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #only thing i have to implement here is that if x is positive, return x
    #else return 0

    #this is the boolean indicating x's sign, 1 if x is positive, 0 if negative
    iden = x > 0
    #simply becomes x*1 or x*0
    out = x * iden

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    iden = x > 0
    dx = dout* iden

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #softmax turns a list of logits into probabilities
    #for this i substract the maximum from each for numerical stability
    #then exp() them
    #then normalize and get the possibilities for each of them
    eps = 1e-15
    #minus maximals of each row
    row_maximals = np.max(x,axis=1,keepdims=True)
    curr_x = x - row_maximals

    #exponentiate them, add them and count probabilities for them and normalize
    exponentas_curr = np.exp(curr_x)
    sums_curr = np.sum(exponentas_curr, axis =1 , keepdims=True)
    albatobebi = exponentas_curr/sums_curr

    #for each, calculate the negative logs of correct class
    #then get the average and that is the loss
    to_size = x.shape[0]  #batch size
    needed_alb = albatobebi[np.arange(to_size), y] #chooses the needed probabilities
    log_alb = np.log(needed_alb+eps) #get logs
    nloss = -np.sum(log_alb) #get -log
    loss = nloss/to_size

    #for getting dx, -1 from probability and then devide by batch size
    #firstly, my dx must be the probabilities i counted beforehand
    dx = albatobebi.copy()
    zusti_class= np.arange(to_size)
    dx[zusti_class,y] = dx[zusti_class,y]-1
    dx = dx/to_size



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #minda mqkonddes mean =0 da variance =1
        #mere davaskalero,davshifto
        const_mean = momentum*running_mean
        const_var = momentum*running_var
        run_momentum = 1-momentum
        #jer davtvli mean da variacias  
        mean_mine = np.mean(x,axis=0)
        var_mine=np.var(x,axis=0)
        curr_mn = var_mine + eps
        mn_maken = np.sqrt(curr_mn)
        gasworebuli_input = (x-mean_mine)/mn_maken
        to_be_wach = gamma*gasworebuli_input

        #esaa shiftingi normalized datasi betati
        out = to_be_wach + beta
        cache= (x,
        gasworebuli_input,
        mean_mine,
        var_mine,
        gamma,
        beta,
        eps
        )


        new_mean_mine = const_mean + run_momentum*mean_mine
        new_var_mine = const_var + run_momentum*var_mine
        #updating axali mnishvnelobebit
        running_mean= new_mean_mine
        running_var = new_var_mine

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #aqac normalizireba minda, ogond sruliad meanis da variaciis
        #nacvlad, gamoviyeneb running_var da running_means
        curr_mr_test = x -running_mean
        curr_test_mn = running_var + eps
        mn_maken = np.sqrt(curr_test_mn)
        input_gamos = curr_mr_test/mn_maken
        tobe_shifted_test= gamma*input_gamos 

        out = tobe_shifted_test +beta
        #testis dros vikideb caches
        cache = None


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #forward passidan chamovitan ragac argumentebs
    #cache-shi mqonda shenaxuli ase
                    #cache= (x,
                    #   gasworebuli_input,
                    #   mean_mine,
                    #    var_mine,
                    #    gamma,
                    #    beta,
                    #    eps
                    #    )

    
    #gamovitan axla argumenteb
    x, gasworebuli_input, mean_mine, var_mine, gamma, beta, eps = cache
    dim1 = x.shape[0]
    dim2 = x.shape[1]


    diff_mean = x - mean_mine
    d_gamma = dout*gamma
    aris_sil = var_mine +eps
    atr_var_mine = np.power(aris_sil, -1.5)
    atr_dev = 1/np.sqrt(aris_sil)
    for_g = dout* gasworebuli_input
    input_n = d_gamma * diff_mean * (-0.5)
    #me mwirdeba davtvalo shemdegi ragacebi:
    # gradienti gamma/beta/normalizirebuli inputisadmi da additional
    #variablebisadmi, romelmac pexi gamoyo batch normalizaciashi
    #gradienti normalizirebisas
    grad_n = d_gamma
    #gradienti gama
    grad_g = np.sum(for_g,axis=0)
    #gradienti beta
    grad_b = np.sum(dout, axis =0)
    #variaciistvis warmoebuli
    der_var = np.sum(input_n * atr_var_mine,axis=0)
    #meanistvis warmoebuli
    term1 = d_gamma* (-atr_dev)
    term2 = -2*diff_mean
    der_mean1 = np.sum(term1,axis=0)
    der_mean2 = der_var* np.sum(term2,axis=0)/dim1
    der_mean = der_mean1 + der_mean2 
    #now updating the dx gvinda 
    dx1 = d_gamma * atr_dev
    dx2= 2*der_var*diff_mean/dim1
    dx3= der_mean/dim1
    dx = dx1+dx2+dx3

    dgamma= grad_g
    dbeta= grad_b
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, gasworebuli_input, mean_mine, var_mine, gamma, beta, eps = cache
    dim1 = x.shape[0]
    aris_sil = var_mine +eps
    root_aris_sil = np.sqrt(aris_sil)
    grn = dout*gamma
    dine_x = dout* gasworebuli_input
    #paqtobrivad igives vaketeb, ogond 1 gamosaxulebashi unda chavteno
    #RATOMGAC yvelaperi
    dbeta = np.sum(dout,axis=0)
    dgamma = np.sum(dine_x,axis=0)
    #gradienti normalizirebulistvis
    new_gradient =grn  

    #1 pormulashi yvelapris sheyra 
    dx = 1.0 / (dim1 * root_aris_sil) * (dim1 * new_gradient - np.sum(new_gradient, axis=0) - gasworebuli_input * np.sum(new_gradient * gasworebuli_input, axis=0))
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    

    #axla damwirdeba rom vipovo mean da variacia, mere normalizireba gavuketo
    #chems datas, scale/shift da mere mexsiereba shevinaxo
    #similar ukve daweril layernorm  forward
    my_mean = np.mean(x,axis=1,keepdims=True)
    my_var = np.var(x,axis=1,keepdims=True)
    var_eps = my_var +eps
    sqr_var_eps=np.sqrt(var_eps)
    changeof_x = x- my_mean

    #normalizireba movvaxdine
    changed_input = changeof_x/sqr_var_eps

    #axla minda scale da shifti, isev gamma scale and beta shift
    to_scale = gamma* changed_input
    out = to_scale + beta

    cache = (
    x,
    changed_input,
    my_mean,
    my_var,
    gamma,
    beta,
    eps
    )
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    #jer wavikitxav argumentebs
    x,gasworebuli_input, mean_mine,var_mine,gamma,beta,eps = cache

    dim1 = x.shape[0]
    dim2=x.shape[1]
    #aq rac unda gavaketo aris vipovo gamma da betas argumenti
    #da aseve normalizirebulis  gradineti
    diff_mean = x - mean_mine
    d_gamma = dout*gamma
    aris_sil = var_mine +eps
    atr_var_mine = np.power(aris_sil, -1.5)
    atr_dev = 1/np.sqrt(aris_sil)
    for_g = dout* gasworebuli_input
    input_n = d_gamma * diff_mean * (-0.5)*atr_var_mine

    #axla daviwyot ushualod datvla
    #gradienti normalizirebuli
    grad_n = d_gamma
    #gradienti gama
    grad_g=np.sum(for_g,axis=0)
    #gradienti beta
    grad_b = np.sum(dout,axis=0)

    der_var = np.sum(input_n, axis=1, keepdims=True)
    #gradient for mean
    term1 = d_gamma * (-atr_dev)
    term2 = -2 * diff_mean/dim2
    der_mean1 = np.sum(term1, axis=1, keepdims=True)
    der_mean2 = der_var * np.sum(term2, axis=1,keepdims=True) 
    der_mean = der_mean1 + der_mean2

    
    dx1 = d_gamma * atr_dev
    dx2 = 2 * der_var * diff_mean / dim2
    dx3 = der_mean / dim2
    dx = dx1 + dx2 + dx3

    #datvlilebi ubralod sworad shevusabame
    dgamma = grad_g
    dbeta = grad_b


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        dim = x.shape
        unit_p = 1.0/p
        #0dan 1mde random ricxviebis generation
        zero_to_one = np.random.rand(*dim)

        #p: Dropout parameter. We keep each neuron output with probability p.
        # ^ esaa mocemuli, amito avdget da 1ianebi mivatyepot yvelgan, romlis
        #albatobac metia pze
        prob_filter = zero_to_one < p

        #chven inverted effects vedzebt ideashi
        final_filtering = prob_filter *unit_p    
        out = x* final_filtering
        mask=final_filtering

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #ak vikideb applying droupout, ubralod input minda
        out=x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout*mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #jer damwirdeba amovigo yvela parametri
    #mere davpaddingo da mere gadavuyve yvelapers
    mr_comp = conv_param['pad']
    mn_comp = conv_param['stride']
    double_padding = 2*mr_comp

    outl_num, chanel_num, filter_h, filter_w = w.shape 
    input_num, pruw, input_h, input_w  = x.shape #pruw shemidzlia davikido, arapershi viyeneb
    mr_hei = input_h +double_padding - filter_h
    mr_wei = input_w + double_padding -filter_w

    #gatenili inputi ikneba chemtis 
    to_ten_input = ((0,0),(0,0),(mr_comp,mr_comp),(mr_comp,mr_comp))
    input_gat = np.pad(x, pad_width=to_ten_input,mode='constant',constant_values=0)
    
    new_dim_1 = 1 + mr_hei // mn_comp #ეს იქნება საზღhვარი 
    new_dim_2= 1+ mr_wei // mn_comp   #ეს იქნება მეორე საზღვარი ჩემთვის 

    #saboloo sruli tensoristvis
    for_tensor = np.zeros((input_num,outl_num,new_dim_1,new_dim_2))
    stances = [(i,j) for i in range(new_dim_1) for j in range (new_dim_2)]

    #axla gadavuyvebi yvelapers stances adgilebze
    for pos_en in range(input_num):
      for pos_ef in range(outl_num):
        const_w = w[pos_ef]
        const_b = b[pos_ef]
        for hor, ver in stances:
          new_hor = hor*mn_comp
          new_ver = ver*mn_comp
          my_new_window = input_gat[pos_en, :, new_hor:new_hor+filter_h, new_ver:new_ver+filter_w]
          sum_counter = my_new_window* const_w
          final_sum = np.sum(sum_counter)
          ans = final_sum+ const_b

            #axla amas chavwer tensorshi
          for_tensor[pos_en,pos_ef,hor,ver] = ans

    out = for_tensor







    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

     # jer amovigot parametrebi da memkvidreoba chashe-dan
    x, w, b, conv_param = cache
    
    # amovigot paddingis da stride-s parametrebi
    mr_comp = conv_param['pad']
    mn_comp = conv_param['stride']
    double_padding = 2*mr_comp
    
    # es parametrebi dzaan mnisvnelovania gradinetis gamosatvalad
    outl_num, chanel_num, filter_h, filter_w = w.shape 
    input_num, na_1, input_h, input_w = x.shape
    
    # dout-dan amovigot gadmocemuli gradienti da gamovtvalot shedegebi
    na_2, na_3, out_h, out_w = dout.shape
    
    # shevikmnat sawyisi gradientebi
    gat_mtl_x = np.zeros((input_num, chanel_num, input_h + 2*mr_comp, input_w + 2*mr_comp))
    warw = np.zeros(w.shape)
    warb = np.zeros(b.shape)
    
    # gatenili inputi mzad unda iyos
    to_ten_input = ((0,0), (0,0), (mr_comp,mr_comp), (mr_comp,mr_comp))
    input_gat = np.pad(x, pad_width=to_ten_input, mode='constant', constant_values=0)
    
    # stances isev gvexmareba gadaviarot yvela pozicia
    stances = [(i,j) for i in range(out_h) for j in range(out_w)]
    
    # gadaviarot yvela magalitze, yvela filterze, da yvela output poziciaze
    for pos_en in range(input_num):
        for pos_ef in range(outl_num):
            # b-s gradient aris dout-is jami
            warb[pos_ef] += np.sum(dout[pos_en, pos_ef])
            
            for hor, ver in stances:
                # gradienti am poziciashi
                grad_here = dout[pos_en, pos_ef, hor, ver]
                
                # gavmotvalot input region es poziciistvis
                new_hor = hor * mn_comp
                new_ver = ver * mn_comp
                my_window = input_gat[pos_en, :, new_hor:new_hor+filter_h, new_ver:new_ver+filter_w]
                
                # dagvchirdeba w-s da x-is gradientebi
                warw[pos_ef] += grad_here * my_window
                gat_mtl_x[pos_en, :, new_hor:new_hor+filter_h, new_ver:new_ver+filter_w] += w[pos_ef] * grad_here
    
    # movachrat padding dx-s
    dx = gat_mtl_x[:, :, mr_comp:mr_comp+input_h, mr_comp:mr_comp+input_w]
    dw = warw
    db = warb

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # jer davitvalot yvela sachiro parametrebi
    input_num, chanel_num, input_h, input_w = x.shape
    
    # pool parametrebis amogeba
    max_height = pool_param['pool_height']
    max_width = pool_param['pool_width']
    mv_comp = pool_param['stride']
    
    # gamomavali parametrebis gamotvla
    out_h = 1 + (input_h - max_height) // mv_comp
    out_w = 1 + (input_w - max_width) // mv_comp
    
    # shedegis tensoristvis adgilis gamoyopa
    for_tensor = np.zeros((input_num, chanel_num, out_h, out_w))
    
    # migvindeba davimaxsovrot max-is poziciebi shebrunebuli svlistvis
    yvelaze_adg = np.zeros((input_num, chanel_num, out_h, out_w, 2), dtype=int)
    
    # shevikmnat output poziciebi
    stances = [(i,j) for i in range(out_h) for j in range(out_w)]
    
    # davitvalot titoeuli ujra
    for pos_en in range(input_num):
        for pos_ef in range(chanel_num):
            for hor, ver in stances:
                # davitvalot sawyisi poziciebi
                saw_h = hor * mv_comp
                saw_w = ver * mv_comp
                
                # amovigot poolingis ubani
                pool_ubani = x[pos_en, pos_ef, saw_h:saw_h+max_height, saw_w:saw_w+max_width]
                for_tensor[pos_en, pos_ef, hor, ver] = np.max(pool_ubani)
                yvelaze_sis = np.unravel_index(np.argmax(pool_ubani), pool_ubani.shape)
                yvelaze_adg[pos_en, pos_ef, hor, ver] = yvelaze_sis
    
    out = for_tensor
                
                

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param, yvelaze_adg)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # jer wamovigot yvela sachiro parametri
    x, pool_param, yvelaze_adg = cache
    
    # movipovot parametrebi saxit da poolistvis
    input_num, chanel_num, input_h, input_w = x.shape
    bigger_x = pool_param['pool_height']
    bigger_w = pool_param['pool_width']
    mv_comp = pool_param['stride']
    
    # davitvalot output-is zomebi (saidan modis dout)
    out_h = 1 + (input_h - bigger_x) // mv_comp
    out_w = 1 + (input_w - bigger_w) // mv_comp
    
    # shevikmnat gradients sawyisi mnishvneloba
    x_unia = np.zeros((input_num, chanel_num, input_h, input_w))
    
    # shevikmnat output poziciebi
    stances = [(i,j) for i in range(out_h) for j in range(out_w)]
    
    # davuvliot yvela magalits, arxs da output pozicias
    for pos_en in range(input_num):
        for pos_ef in range(chanel_num):
            for hor, ver in stances:
                # davitvalot sawyisi poziciebi poolingis ubnistvis
                saw_h = hor * mv_comp
                saw_w = ver * mv_comp
                
                # wamovigot maqsimaluris pozicia am ubnistvis
                max_h, max_w = yvelaze_adg[pos_en, pos_ef, hor, ver]
                
                # gradienti gadis mxolod maqsimalur mnishvnelobaze
                x_unia[pos_en, pos_ef, saw_h + max_h, saw_w + max_w] += dout[pos_en, pos_ef, hor, ver]
    
    dx = x_unia

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #jer mwirdeba monacemebis ageba, mere shapingi da batch normalizacia
    #romelic ukve implemented aris
    coords = x.shape
    #mwirdeba batch size, channels, width/height
    #amitom avigeb magat coords-dan
    bath_z, chan_z , h_size, w_size = coords
    real_b_size = bath_z*h_size*w_size
    #jer davaprtyeleb 
    brty_x = x.transpose(0, 2, 3, 1)
    final_x = brty_x.reshape(real_b_size,chan_z)
    #mwirdeba normalizacia, romelicaa batchnorm_forward
    my_output = batchnorm_forward(final_x,gamma,beta, bn_param)
    to_trans, mem = my_output

    out_before_trans = to_trans.reshape(bath_z, h_size, w_size, chan_z)
    final_outing = out_before_trans.transpose(0,3,1,2)

    out = final_outing
    cache = (mem, coords)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #aqac dimensionebbi unda avigo, 1 batchnorm_backwars minda 
    #da mere isev ukan wamogeba 
     #jer mwirdeba monacemebis ageba, mere shapingi da batch normalizacia
    #romelic ukve implemented aris
    mem, coords = cache
    
    #mwirdeba batch size, channels, width/height
    #amitom avigeb magat coords-dan
    bath_z, chan_z, h_size, w_size = coords
    real_b_size = bath_z*h_size*w_size
    #jer davaprtyeleb dout-s
    brty_dout = dout.transpose(0, 2, 3, 1)
    final_dout = brty_dout.reshape(real_b_size, chan_z)
    #mwirdeba batchnorm backward, romelic ukve implementirebulia
    my_output = batchnorm_backward(final_dout, mem)

    new_additional_x, change_gamma, change_beta = my_output
    out_before_trans = new_additional_x.reshape(bath_z, h_size, w_size, chan_z)
    final_outing = out_before_trans.transpose(0, 3, 1, 2)

    dx = final_outing
    dgamma = change_gamma
    dbeta = change_beta
        

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    axis_saver= (2,3,4)
    #ჯერ isev avigeb parametrebs
    params = x.shape
    bath_z, arxeb_z, h_size, w_size = params
    const_to_keep = arxeb_z // G
    eps_mine = gn_param.get("eps",1e-5)

    #mwirdeba, rom davamushaovo chemi input
    #gavuketeb reshapings 
    changed_input = x.reshape(bath_z,G,const_to_keep,h_size,w_size)
    #axla mwirdeba variacia da mean 
    my_var = np.var(changed_input, axis = axis_saver,keepdims=True )
    my_mean = np.mean(changed_input,axis= axis_saver,keepdims=True)
    eps_var= eps_mine + my_var
    sq_eps_var = np.sqrt(eps_var)
    x_shifted = changed_input -my_mean


    #wina punqciebis msgavsad, jer minda normalization, mere shiftingi 
    #gammaze gamravlebis mere 
    # RESHAPINGIA SAWIRO !!!!!!!!!!!!!!!!!!!!!!!! 
    dal_input = x_shifted/sq_eps_var
    final_out_br = dal_input.reshape(bath_z,arxeb_z, h_size, w_size)
    gamma_gasw = gamma* final_out_br

    #amis mere axla mwirdeba gamma/beta
    final_out = gamma_gasw +beta

    mem_out =(
      x,
      final_out_br,
      my_mean,
      my_var,
      gamma,
      beta,
      G, 
      eps_mine
    )

    out = final_out
    cache=mem_out


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    axis_saving = (0,2,3)
    axis_saving_2 = (2,3,4)
    pw=-1.5
    #jer vigeb marto sawiro argumentebs
    out1=cache
    x, normal_x, my_mean, my_var, gamma, beta, G, eps_mine = out1
    out2=x.shape
    eps_mine = eps_mine * 1.0001
    bath_z, arxeb_z, h_size, w_size = out2
    dn_x_input = dout * normal_x 
    const_shes = arxeb_z//G
    areal_mine = h_size*w_size

    kv = gamma.reshape(1, arxeb_z, 1, 1)
    var_epsi = my_var +eps_mine
    sqrt_var_epsi = np.sqrt(var_epsi)
    reverse_mn = -1 / sqrt_var_epsi
    all_amount_combined = const_shes * areal_mine
    #axla mwirdeba, rom am monacemebit davtvalo gradientebi
    gamma_change = np.sum(dout * normal_x, axis=axis_saving, keepdims=True)
    beta_change = np.sum(dout, axis=axis_saving, keepdims=True)
    gradient_nx =dout * kv

    reshaped_nx = gradient_nx.reshape(bath_z,G, const_shes, h_size,w_size)
    input_ch = x.reshape(bath_z,G, const_shes,h_size,w_size)
    difference_mean_x = input_ch - my_mean

    var_epsi_changed = np.power(var_epsi,pw)
    g_const = (-0.5)*var_epsi_changed
    const_xar = reshaped_nx * difference_mean_x
    change_var = np.sum(const_xar* g_const,axis = axis_saving_2,keepdims=True) #variaciistvis
    d_ch_var = 2* change_var
    #igive unda vqnat meanistvis 

    mean_nawili1 = reshaped_nx * reverse_mn
    mean_nawili2 = -2 * difference_mean_x
    mc_mr =  change_var * np.sum(mean_nawili2, axis=axis_saving_2, keepdims=True)
    mean_jami1 = np.sum(mean_nawili1, axis=axis_saving_2, keepdims=True)
    mean_jami2 = mc_mr/ all_amount_combined
    mean_tot = mean_jami1 + mean_jami2


    #axla unda davamushavo msgavsadve x 
    x_pt1 = reshaped_nx/sqrt_var_epsi
    x_pt2 = d_ch_var * difference_mean_x / all_amount_combined
    x_pt3 = mean_tot/all_amount_combined
    change_x_tot = x_pt1+x_pt2+x_pt3

    #bolos ubralod reshaping gvinda da egaa
    tot_x_fin = change_x_tot.reshape(bath_z, arxeb_z, h_size, w_size)
    
    dgamma = gamma_change
    dbeta = beta_change
    dx = tot_x_fin
    

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
