import torch
from tqdm import trange

import pymc as pm
import pytensor as pt
from pymc import PolyaGamma as PG


def _compute_mean_a(v, b, theta, w, y):
    S = torch.sum(torch.mul(theta, y-0.5-torch.mul(b, w)))
    mean = v*S
    return mean.item()


def _compute_var_a(sigmasq, theta, w):
    S = torch.sum(torch.mul(w, torch.pow(theta, 2))) + 1/sigmasq
    var = 1/S
    return var.item()


def _compute_mean_b(v, a, theta, w, y):
    S = torch.sum(y-0.5-a*torch.mul(theta, w))
    mean = v*S
    return mean.item()


def _compute_var_b(sigmasq, w):
    S = torch.sum(w) + 1/sigmasq
    var = 1/S
    return var.item()


def _compute_mean_t(v, a, b, w, y):    
    S = torch.sum(torch.mul(a, y-0.5-torch.mul(b, w)))
    mean = v*S
    return mean.item()    


def _compute_var_t(sigmasq, w, a):
    S = torch.sum(torch.mul(w, torch.pow(a, 2))) + 1/sigmasq
    var = 1/S
    return var.item()


def polyagamma(init_a, init_b, init_t, init_w, y, sigmasq_a, sigmasq_b, sigmasq_t, niter=10000):

    assert init_a.shape == init_b.shape
    assert init_t.shape[0] == y.shape[0]
    assert y.shape == init_w.shape
    
    I = len(init_a)
    P = len(init_t)
    
    # samples
    A = torch.empty(size=(niter, I))
    B = torch.empty(size=(niter, I))
    THETA = torch.empty(size=(niter, P))
    W = torch.empty(size=(niter, P, I))
    
    A[0] = init_a
    B[0] = init_b
    THETA[0] = init_t
    W[0] = init_w

    print(f"Starting Gibbs sampler for Polya-Gamma... \n--------------------------------------------\n")
    for s in trange(1, niter):

        # calculate scale values for PG draw
        Z = torch.abs(torch.mul(torch.t(THETA[s-1, None]), A[s-1, None]) + B[s-1])  
        assert Z.shape == (P, I)
        
        # sample Ws
        W[s] = torch.from_numpy(pm.draw(PG.dist(h=1, z=Z)))

        # sample As
        for i in range(I):
            var_a = _compute_var_a(sigmasq_a, THETA[s-1], W[s][:,i])
            mean_a = _compute_mean_a(var_a, B[s-1][i], THETA[s-1], W[s][:,i], y[:,i])
            A[s][i] = torch.distributions.Normal(mean_a, var_a).sample()

        # sample Bs
        for i in range(I):
            var_b = _compute_var_b(sigmasq_b, W[s][:,i])
            mean_b = _compute_mean_b(var_b, A[s][i], THETA[s-1], W[s][:,i], y[:,i])
            B[s][i] = torch.distributions.Normal(mean_b, var_b).sample()

        # sample THETAs
        for p in range(P):
            var_t = _compute_var_t(sigmasq_t, W[s][p], A[s])
            mean_t = _compute_mean_t(var_t, A[s], B[s], W[s][p,:], y[p,:])
            THETA[s][p] = torch.distributions.Normal(mean_t, var_t).sample()

    return A, B, THETA, W

