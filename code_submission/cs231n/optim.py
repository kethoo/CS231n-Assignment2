import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #need wina velocitiues, make vector da mere shevteno shignit axali
    #velocity da mere shevteno eg konfiguraciashi
    curr_rate = config['learning_rate']
    #gradienti maqvs da mwirdeba momentum
    mu = config['momentum']

    #update formula v-stvis
    muv = mu*v
    rate_gr = curr_rate*dw
    v = muv - rate_gr

    #axla shevtenot chvenin v configshi
    #config['velocity'] = v

    #wonebis update v-s gamoyenebit
    next_w = v+w

    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #mokled, ak mwirdeba rom caches davakvirde da sworad shevinaxo
    #da update unda gavuketo sworad

    #ak mnishvnelovania moving average of sqrd GRADIENTS !!!!!!
    #EPSILONI AR GAMOMRCHES
    eps = config['epsilon']
    #jer load argumentebi, romelic pormulistvis damwirdeba
    curr_rate, lp_rate = config['learning_rate'],config['decay_rate']
    mem = config['cache']

    changeof_rate = (dw*dw)
    second_seg = (1-lp_rate)*changeof_rate
    mem = lp_rate*mem +  second_seg
    config['cache']=mem #axali mem ukan chavteno configurationshi
    mem_mn = np.sqrt(mem)
    mem_eps = mem_mn+eps
    l_gradient = curr_rate* dw


    #axla update the weights 
    next_w = w -l_gradient/mem_eps



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #jer gvwirdeba momentum pirveli (m) da momentum meore(v)
    #gvwirdeba aseve beta1, beta2, epsiloni(0ze gayopisgan gvicavs)
    #gradienti 
    

    #tze tviton gveubmneba, rom sanam amovigebt,ikamde unda shevcvalot
    before_mod_t = config['t']
    config['t']=before_mod_t+1
    #axla avigot t da danarcheni cvladebic
    m,v,t=config['m'],config['v'],config['t']
    current_rate = config['learning_rate']
    epsilon=config['epsilon']
    beta1, beta2 = config['beta1'], config['beta2']

    #updating momentums
    av_m = beta1*m +(1-beta1)*dw
    av_v  = beta2*v +(1-beta2)*(dw*dw)

    m=av_m
    v=av_v
    #es axlebi isev configshi unda shevteno, sworadaaa dasabrunebeli
    config['v'] =v
    config['m']=m

    new_beta1= beta1**t
    new_beta2=beta2**t
    mn1 = 1-new_beta1
    mn2 = 1- new_beta2
    #bias daumate for correction
    swr_m = m/mn1
    swr_v= v/mn2

    final_mr= current_rate*swr_m
    final_swr_v = np.sqrt(swr_v)
    final_mn= epsilon + final_swr_v
    transfering_pard = final_mr/final_mn
    #sabolood, unda shevcvalo weights
    next_w = w - transfering_pard

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config
