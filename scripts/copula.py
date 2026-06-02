import pyvinecopulib as pv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import norm

def sort(data,n,x1='angle_x_deg',x2='angle_y_deg'):
    """  
    Sort pd dataframe by taking 2 parameters for a particular z_idx value and convert to numpy
    """
    return data[data['z_idx'] == n][[x1,x2]].to_numpy()

def bivariate_copula(data,n,model=None): #n is number of fibers in a layer
    """ 
    Set up copula from bivariate dataset
    Returns synthetic dataset and corresponding copula
    """
    # Perform PIT on observed data
    u = pv.to_pseudo_obs(data)

    # If no family is specified, this means that cop.select will run to choose a family
    # You should avoid running without specifying a family because it takes ages to run
    if model:
        cop = pv.Bicop(model)
        cop.fit(data=u)
    else:
        cop = pv.Bicop()
        cop.select(data=u)

    # Create synthetic dataset from the fresh copula
    u_sim = cop.simulate(n)
    
    # Perform inverse PIT
    data_sim = np.asarray([np.quantile(data[:, i], u_sim[:, i]) for i in range(0, 2)])
    data_sim = np.transpose(data_sim)

    return data_sim,cop

def bivar_cop_u(data,n):
     # Perform PIT on observed data
    u = pv.to_pseudo_obs(data)

    # If no family is specified, this means that cop.select will run to choose a family
    # You should avoid running without specifying a family because it takes ages to run
    cop = pv.Bicop(pv.student)
    cop.fit(data=u)

    # Create synthetic dataset from the fresh copula
    u_sim = cop.simulate(n)
    return u_sim, cop

def depth_mem(data, u, rho):
    z1, z2 = norm.ppf(u[0]), norm.ppf(u[1]) #Transform from u-space to z-space
    #Depth-memory calculation
    z2_x = z1[:,0] * rho[0] + z2[:,0] * (1-rho[0]**2) ** 0.5
    z2_y = z1[:,1] * rho[1] + z2[:,1] * (1-rho[1]**2) ** 0.5
    z2 = np.concatenate([np.reshape(z2_x, (-1, 1)), np.reshape(z2_y, (-1, 1))], axis=1) #Reformat to a single array of (x, y) tilts
    u2 = norm.cdf(z2) #Transform from z-space to u-space
    data_sim = np.asarray([np.quantile(data[:, i], u2[:, i]) for i in range(0, 2)]) #Transform from u-space to x-space
    data_sim = np.transpose(data_sim) #Reshape
    return data_sim, u2



def reconstruct(df_clean,arr_sim,zz_complete,n_fibers,par_1,par_2):
    """ 
    Reconstruct synthetic fibers from initial starting points and dx and dy arrays
    Stack data in the same manner that it was in clean_df
    """
    z_scale = 500 / 128

    # Take base layer of measured fibers
    df_0 = sort(df_clean,0,'x','y')

    # Broadcast the starting points in an array which has the same shape as the angles array
    # This makes it possible to use np operations to add the angles 
    sim_fibers = np.broadcast_to(df_0,(len(zz_complete),n_fibers,2)).copy()


    # Cummulatively sum the dx and dy values anb convert angles to distance
    sim_fibers[1:] += np.cumsum(np.tan(np.radians(arr_sim[1:]))*z_scale,axis=0)


    # Stack the arrays to get data back in original shape
    stacked_sim_fibers = np.vstack(sim_fibers)
    
    # Create array in the format 0,1,2,...,n_fibers,0,1,2,...,n_fibers to repeat as many times as there are layers
    fibre_id_arr = np.tile(np.arange(n_fibers),len(zz_complete))
    # Create array in the format 0,0,0...,1,1,1,1..2,2,2,2 where each numbers repeats as many times as there are fibers
    zz_arr = np.repeat(zz_complete,n_fibers)

    # df_columns = ['fibre_id','z_idx','x','y']

    sim_df = pd.DataFrame()
    sim_df[par_1] = arr_sim[:,:,0].flatten()
    sim_df[par_2] = arr_sim[:,:,1].flatten()
    sim_df['x'] = stacked_sim_fibers[:,0]
    sim_df['y'] = stacked_sim_fibers[:,1]
    sim_df['z_idx'] = zz_arr
    sim_df['fibre_id'] = fibre_id_arr
    sim_df['z'] = sim_df['z_idx'] * z_scale

    return sim_df