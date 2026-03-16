import pyvinecopulib as pv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# np.random.seed(0)  # seed for the random generator
# n = 1000  # number of observations
# d = 3  # the dimension
# mean = 1 + np.random.normal(size=d)  # mean vector
# cov = np.random.normal(size=(d, d))  # covariance matrix
# cov = np.dot(cov.transpose(), cov)  # make it non-negative definite
# x = np.random.multivariate_normal(mean, cov, n)

def sort(data,n,x1='angle_x_deg',x2='angle_y_deg'):
    # Filter out rows which do not match our z value
    df_z = data[data['z'] == n]
    # take out dx and dy rows  
    df_z = df_z[[x1,x2]]
    x = df_z.to_numpy()
    # print(x)
    return x

def bivariate_copula(data,n,family=False): #n is number of fibers in a layer
    u = pv.to_pseudo_obs(data)
    # pv.pairs_copula_data(u, scatter_size=0.5)

    # If no family is specified, this means that cop.select will run to choose a family
    # You should avoid running without specifying a family because it takes ages to run
    if family:
        cop = pv.Bicop(family)
    else:
        cop = pv.Bicop()
        cop.select(data=u)

    cop.fit(data=u)

    #print(cop)
    # cop.plot()

    n_sim = n
    u_sim = cop.simulate(n_sim, seeds=[1, 2, 3, 4])
    data_sim = np.asarray([np.quantile(data[:, i], u_sim[:, i]) for i in range(0, 2)])
    data_sim = np.transpose(data_sim)
    return data_sim,cop

def vine_copula(x,n): #n is number of fibers in a layer
    # Do PIT
    u = pv.to_pseudo_obs(x)

    # automatically fit best copula model
    cop = pv.Vinecop.from_data(data=u)

    # What does this do??
    pv.pairs_copula_data(u, scatter_size=0.5)


    
    u_sim = cop.simulate(n, seeds=[1, 2, 3, 4])

    # Reverse pit to get actual values
    data_sim = np.asarray([np.quantile(x[:, i], u_sim[:, i]) for i in range(0, 2)])
    data_sim = np.transpose(data_sim)
    return data_sim


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

