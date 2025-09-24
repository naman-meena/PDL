# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 13:24:35 2025

@author: naman

quick primal-dual experiment
"""

import numpy as np
import matplotlib.pyplot as plt

# objective
def f(theta):
    return theta[0]**2 + theta[1]**2

# constraint g(x)=0
def g(theta):
    return theta[0] + theta[1] - 1

# grads
def df(theta):
    return np.array([2*theta[0], 2*theta[1]])

def dg(theta):
    return np.array([1., 1.])

# primal-dual loop
def primal_dual():
    K, L = 50, 5
    rho, alpha, rho_max = 1., 1.1, 100
    lr = 0.005

    theta = np.array([2., -1.])
    lam = 0.0

    hist = {"theta":[], "phi":[], "obj":[], "viol":[]}

    for k in range(K):
        for _ in range(L):
            gt = g(theta)
            gradL = df(theta) + lam*dg(theta) + rho*gt*dg(theta)
            theta = theta - lr*gradL

        v = g(theta)
        lam = lam + rho*v
        if rho < rho_max: rho *= alpha

        hist["theta"].append(theta.copy())
        hist["phi"].append(lam)
        hist["obj"].append(f(theta))
        hist["viol"].append(abs(v))

    print("done. theta=",theta," lambda=",lam)
    return hist

# run
hist = primal_dual()

# plots
thetas = np.array(hist["theta"])
phis = np.array(hist["phi"])

fig,ax = plt.subplots(2,2,figsize=(12,9))

ax[0,0].plot(thetas[:,0],label="x1")
ax[0,0].plot(thetas[:,1],label="x2")
ax[0,0].axhline(0.5,color="r",ls="--")
ax[0,0].set_title("theta")
ax[0,0].legend(); ax[0,0].grid()

ax[0,1].plot(phis,label="lambda")
ax[0,1].axhline(-1,color="r",ls="--")
ax[0,1].set_title("phi")
ax[0,1].legend(); ax[0,1].grid()

ax[1,0].plot(hist["obj"],label="obj")
ax[1,0].axhline(0.5,color="r",ls="--")
ax[1,0].set_title("objective")
ax[1,0].legend(); ax[1,0].grid()

ax[1,1].plot(hist["viol"],label="|g|")
ax[1,1].set_yscale("log")
ax[1,1].set_title("constraint violation")
ax[1,1].legend(); ax[1,1].grid()

plt.tight_layout()
plt.show()
