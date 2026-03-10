import pyvinecopulib as pv
import numpy as np
import pandas as pd
import matplotlib as plt

np.random.seed(0)  # seed for the random generator
n = 1000  # number of observations
d = 3  # the dimension
mean = 1 + np.random.normal(size=d)  # mean vector
cov = np.random.normal(size=(d, d))  # covariance matrix
cov = np.dot(cov.transpose(), cov)  # make it non-negative definite
x = np.random.multivariate_normal(mean, cov, n)


def copula_model(data,n): #n is number of fibers in a layer
    u = pv.to_pseudo_obs(data)
    pv.pairs_copula_data(u, scatter_size=0.5)
    cop = pv.Vinecop.from_data(data=u)
    print(cop)
    #cop.plot()

    n_sim = n
    u_sim = cop.simulate(n_sim, seeds=[1, 2, 3, 4])
    data_sim = np.asarray([np.quantile(x[:, i], u_sim[:, i]) for i in range(0, d)])
    data_sim = np.transpose(data_sim)
    return data_sim

data_sim = copula_model(x,1000)
print(np.mean(x), np.std(x))
print(np.mean(data_sim), np.std(data_sim))

def coordinates(layer, data_sim, deltaz):
    # newlayer = np.empty(len(data_sim),2)
    newlayer = []

    for i in range(len(data_sim)):
        x = layer[i][0] + deltaz / np.tan(data_sim[i][0])
        y = layer[i][1] + deltaz / np.tan(data_sim[i][1])
        
        # newlayer[i,0] = x
        # newlayer[i,1] = y
        
        newlayer.append([x,y])

    return newlayer