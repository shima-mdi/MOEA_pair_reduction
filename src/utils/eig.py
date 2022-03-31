import numpy as np
from scipy import linalg
import numpy.polynomial.hermite as herm
import sys
import math


def run_modeling_Bradley_Terry(alpha):
    M, M_ = alpha.shape
    assert M == M_

    iteration = 0
    p = 1.0 / M * np.ones(M)
    change = sys.float_info.max

    DELTA_THR = 1e-8

    while change > DELTA_THR:
        iteration += 1
        p_prev = p
        n = alpha + alpha.T
        pp = np.tile(p, (M, 1)) + np.tile(p, (M, 1)).T
        p = np.sum(alpha, axis=1) / np.sum(n / pp, axis=1)

        p = p / np.sum(p)

        change = linalg.norm(p - p_prev)

    n = alpha + alpha.T
    pp = np.tile(p, (M, 1)) + np.tile(p, (M, 1)).T
    lbda_ii = np.sum(-alpha / np.tile(p, (M, 1))**2 + n / pp**2, axis=1)

    lbda_ij = n / pp*2
    lbda = lbda_ij + np.diag(lbda_ii)
    cova = np.linalg.pinv(np.vstack([np.hstack(
        [-lbda, np.ones([M, 1])]), np.hstack([np.ones([1, M]), np.array([[0]])])]))
    vari = np.diagonal(cova)[:-1]
    # stdv = np.sqrt(vari)

    scores = np.log(p)
    # scores_std = stdv / p # y = log(x) -> dy = 1/x * dx

    return scores, cova[:-1, :-1]

def EIG_GaussianHermitte_matrix_Hybrid_MST(mu_mtx, sigma_mtx):
    """ this is the matrix implementation version"""
    """mu is the matrix of difference of two means (si-sj), sigma is the matrix of sigma of si-sj"""
    epsilon = 1e-9
    M, M_ = np.shape(mu_mtx)

    mu = np.reshape(mu_mtx, (1, -1))
    sigma = np.reshape(sigma_mtx, (1, -1))

    def fs1(x): return (1./(1+np.exp(-np.sqrt(2)*sigma*x-mu))) * \
        (-np.log(1+np.exp(-np.sqrt(2)*sigma*x-mu)))/np.sqrt(math.pi)

    def fs2(x): return (1-1./(1+np.exp(-np.sqrt(2)*sigma*x-mu))) * \
        (np.log(np.exp(-np.sqrt(2)*sigma*x-mu) /
         (1+np.exp(-np.sqrt(2)*sigma*x-mu))))/np.sqrt(math.pi)

    def fs3(x): return 1./(1+np.exp(-np.sqrt(2)*sigma*x-mu))/np.sqrt(math.pi)
    def fs4(x): return (1-1./(1+np.exp(-np.sqrt(2)*sigma*x-mu)))/np.sqrt(math.pi)

    x, w = herm.hermgauss(30)
    x = np.reshape(x, (-1, 1))
    w = np.reshape(w, (-1, 1))

    es1 = np.sum(w*fs1(x), 0)
    es2 = np.sum(w*fs2(x), 0)
    es3 = np.sum(w*fs3(x), 0)
    es3 = es3*np.log(es3+epsilon)
    es4 = np.sum(w*fs4(x), 0)
    es4 = es4*np.log(es4+epsilon)

    ret = es1 + es2 - es3 + es4
    ret = np.reshape(ret, (M, M_))

    ret = -np.triu(ret, 1)
    info = ret[np.triu_indices(ret.shape[0], k=1)]
    return info


def EIG(mu, mu_cova):
    pvs_num = len(mu)

    eig = np.zeros((pvs_num, pvs_num))
    mu_1 = np.tile(mu, (pvs_num, 1))

    sigma = np.diag(mu_cova)
    sigma_1 = np.tile(sigma, (pvs_num, 1))

    mu_diff = mu_1.T-mu_1
    sigma_diff = np.sqrt(abs(sigma_1.T+sigma_1-2*mu_cova))
    eig = EIG_GaussianHermitte_matrix_Hybrid_MST(mu_diff, sigma_diff)
    return eig