import torch
from tqdm import trange

torch.manual_seed(0)


def loga_conditional(a, b, theta, y, sigmasq):
    """Function to compute the (log) full conditional probability of an observed value of 
    variable a at a given index i. To get the actual conditional probability, raise
    Euler's constant to the power of the returned value.
    
    Parameters
    ----------
    a : torch.tensor
        The observed value of a at index i
    b : torch.tensor
        The observed value of b at index i
    theta : torch.tensor
        A vector of observed theta values
    y : torch.tensor
        The ith column vector of the data Y
    sigmasq : float
        The variance of variable a
        
    Returns
    -------
    prob : torch.tensor
        The (log) conditional probability
    """
    
    assert theta.shape[0] == y.shape[0]
    
    logsum = 0
    for p in range(y.shape[0]):
        logsum += (a * y[p] * theta[p]) - torch.log(1 + torch.exp(a*theta[p]+b))
    
    prob = logsum - (torch.pow(a, 2)/(2*sigmasq))
    return prob


def logb_conditional(b, a, theta, y, sigmasq):
    """Function to compute the (log) full conditional probability of an observed value of 
    variable b at a given index i. To get the actual conditional probability, raise
    Euler's constant to the power of the returned value.
    
    Parameters
    ----------
    a : torch.tensor
        The observed value of a at index i
    b : torch.tensor
        The observed value of b at index i
    theta : torch.tensor
        A vector of observed theta values
    y : torch.tensor
        The ith column vector of the data Y
    sigmasq : float
        The variance of variable a
        
    Returns
    -------
    prob : torch.tensor
        The (log) conditional probability
    """
    
    assert theta.shape[0] == y.shape[0]
    
    logsum = 0
    for p in range(y.shape[0]):
        logsum += (b * y[p]) - torch.log(1 + torch.exp(a*theta[p]+b))  
    
    prob = logsum - (torch.pow(b, 2)/(2*sigmasq))
    return prob


def logtheta_conditional(theta, a, b, y, sigmasq):
    """Function to compute the (log) full conditional probability of an observed value of 
    variable theta at a given index p. To get the actual conditional probability, raise
    Euler's constant to the power of the returned value.
    
    Parameters
    ----------
    a : torch.tensor
        A vector of observed a values
    b : torch.tensor
        A vector of observed b values
    theta : torch.tensor
        The observed value of theta at index p
    y : torch.tensor
        The pth row vector of the data Y
    sigmasq : float
        The variance of variable theta
        
    Returns
    -------
    prob : torch.tensor
        The (log) conditional probability
    """

    assert a.shape == b.shape
    
    logsum = 0
    for i in range(a.shape[0]):
        logsum += (a[i] * y[i] * theta) - torch.log(1 + torch.exp(a[i]*theta + b[i]))
    
    prob = logsum - (torch.pow(theta, 2)/2*sigmasq)
    return prob


def metropolis(prev_sample, sigmasq, avg_acc, full_cond, *full_cond_args):
    """Function to perform random-walk metropolis sampling of a full conditional distribution. 

    Parameters
    ----------
    prev_sample : torch.Tensor
        Previous sampled state, to update or not
    sigmasq : float
        The variance of the random walk
    avg_acc : torch.Tensor
        The running average acceptance rate for the samples
    full_cond : function
        The full conditional distribution from which to sample
    *full_cond_args : tuple
        Additional arguments needed for the full conditional distribution, if any

    Returns
    -------
    update : torch.Tensor
        The next state of the random walk
    """
    
    prop = prev_sample + (torch.distributions.Normal(torch.tensor([0.0]), sigmasq)).sample()
    
    # Ensure that values are within range [0,1]
    if (prop > 1) or (prop < 0):
        A = 0
    else:
        logr = full_cond(prop, *full_cond_args) - full_cond(prev_sample, *full_cond_args)    
        A = torch.min(torch.tensor([1.0]), torch.exp(logr)) 
    #logr = full_cond(prop, *full_cond_args) - full_cond(prev_sample, *full_cond_args)    
    #A = torch.min(torch.tensor([1.0]), torch.exp(logr)) 
        
    U = (torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))).sample()

    update = None
    
    if U <= A:
        update = prop
    else:
        update = prev_sample

    acc_diff = A - avg_acc
    return update, acc_diff


def ada_metropolis(prev_sample, sigmasq, scale, s, avg_acc, full_cond, *full_cond_args, rho=0.6, tau=0.3):
    """Function to perform adaptive random-walk metropolis sampling of a full conditional distribution. 

    Parameters
    ----------
    prev_sample : torch.Tensor
        Previous sampled state, to update or not
    sigmasq : float
        The variance of the random walk
    scale : torch.Tensor
        The scale value for adaptation
    s : int
        Current iteration
    avg_acc : torch.Tensor
        The running average acceptance rate for the samples
    full_cond : function
        The full conditional distribution from which to sample
    *full_cond_args : tuple
        Additional arguments needed for the full conditional distribution, if any

    Returns
    -------
    update : torch.Tensor
        The next state of the random walk
    scale : torch.Tensor
        The updated scale value for adaptation
    acc_diff : torch.Tensor
        A sample-specific value to calculate overall average acceptance rates
    """

    #print(scale.shape)
    prop = prev_sample + (torch.exp(scale)*torch.distributions.Normal(torch.tensor([0.0]), sigmasq).sample())

    # Ensure that sampled values are within [0,1]
    if (prop > 1) or (prop < 0):
        A = 0
    else:
        logr = full_cond(prop, *full_cond_args) - full_cond(prev_sample, *full_cond_args)    
        A = torch.min(torch.tensor([1.0]), torch.exp(logr)) 
    
    #logr = full_cond(prop, *full_cond_args) - full_cond(prev_sample, *full_cond_args)    
    #A = torch.min(torch.tensor([1.0]), torch.exp(logr)) 
    U = (torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))).sample()

    update = None
    
    if U <= A:
        update = prop
    else:
        update = prev_sample

    scale = scale + (1/s**rho)*(A-tau)
    acc_diff = A - avg_acc
    return update, scale, acc_diff


def gibbs(init_a, init_b, init_theta, y, sigmasq_a, sigmasq_b, sigmasq_t, niter=10000, adapt=False):
    """Function to perform Gibbs sampling, using Metropolis-within-Gibbs for the intractable full conditional distributions.

    Parameters
    ----------
    init_a : torch.Tensor
        The initial state of variable a
    init_b : torch.Tensor
        The initial state of variable b
    init_theta : torch.Tensor
        The initial state of variable theta
    y : torch.Tensor
        The data
    sigmasq_a : torch.Tensor
        The prior for the variance of variable a
    sigmasq_b : torch.Tensor
        The prior for the variance of variable b
    sigmasq_t : torch.Tensor
        The prior for the variance of variable theta
    niter : int
        The number of samples to collect

    Returns
    -------
    A : torch.Tensor
        The samples for each element of variable a
    B : torch.Tensor
        The samples for each element of variable b
    THETA : torch.Tensor
        The samples for each element of variable theta
    """
    
    assert init_a.shape == init_b.shape
    assert init_theta.shape[0] == y.shape[0]
    
    I = len(init_a)
    P = len(init_theta)
    
    # samples
    A = torch.empty(size=(niter, I))
    B = torch.empty(size=(niter, I))
    THETA = torch.empty(size=(niter, P))

    # Average acceptance probabilities
    avg_acc_a, avg_acc_b, avg_acc_t = torch.zeros((I,)), torch.zeros((I,)), torch.zeros((P,))
    
    A[0] = init_a
    B[0] = init_b
    THETA[0] = init_theta
    
    if adapt:
        scale_a, scale_b, scale_t = torch.full((I,), -0.2), torch.full((I,), -0.2), torch.full((P,), -0.2)
        print(f"Starting Gibbs sampler with adaptation... \n--------------------------------------------\n")
    else:
        print(f"Starting Gibbs sampler... \n--------------------------------------------\n")
        
    for s in trange(1, niter):

        for i in range(I):
            if not adapt:
                A[s][i], ad = metropolis(A[s-1][i], sigmasq_a, avg_acc_a[i], loga_conditional, B[s-1][i], THETA[s-1], y[:,i], sigmasq_a)
                avg_acc_a[i] = avg_acc_a[i] + (1/s)*ad
            else:
                A[s][i], sc, ad = ada_metropolis(A[s-1][i], sigmasq_a, scale_a[i], s, avg_acc_a[i], 
                                                 loga_conditional, B[s-1][i], THETA[s-1], y[:,i], sigmasq_a)
                avg_acc_a[i] = avg_acc_a[i] + (1/s)*ad
                scale_a[i] = sc
                
        for i in range(I):
            if not adapt:
                B[s][i], ad = metropolis(B[s-1][i], sigmasq_b, avg_acc_b[i], logb_conditional, A[s][i], THETA[s-1], y[:,i], sigmasq_b)
                avg_acc_b[i] = avg_acc_b[i] + (1/s)*ad
            else:
                B[s][i], scale_b[i], ad = ada_metropolis(B[s-1][i], sigmasq_b, scale_b[i], s, avg_acc_b[i], 
                                                         logb_conditional, A[s][i], THETA[s-1], y[:,i], sigmasq_b)
                avg_acc_b[i] = avg_acc_b[i] + (1/s)*ad
                
        for p in range(P):
            if not adapt:
                THETA[s][p], ad = metropolis(THETA[s-1][p], sigmasq_t, avg_acc_t[p], logtheta_conditional, A[s], B[s], y[p,:], sigmasq_t)
                avg_acc_t[p] = avg_acc_t[p] + (1/s)*ad 
            else:
                THETA[s][p], scale_t[p], ad = ada_metropolis(THETA[s-1][p], sigmasq_t, scale_t[p], s, avg_acc_t[p], 
                                                             logtheta_conditional, A[s], B[s], y[p,:], sigmasq_t)
                avg_acc_t[p] = avg_acc_t[p] + (1/s)*ad  
                
    print("Done sampling.")
    print("Average acceptance rates:\n", avg_acc_a, "\n", avg_acc_b, "\n", avg_acc_t) 
    return A, B, THETA

