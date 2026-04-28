import pyvinecopulib as pv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare

def sort(data,n,x1='angle_x_deg',x2='angle_y_deg'):
    """  
    Sort pd dataframe by taking 2 parameters for a particular z_idx value
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


def plot_cop_parameters(cop_lst,ax1,ax2):
    """ 
    Plot copula parameters as a function of Z
    Put in ax1 and ax2 to be able to plot parameters for different copula collections
    """

    zz = np.arange(len(cop_lst))
    model_lst = [cop.family for cop in cop_lst]

    # Check if all the copulas are of the same family, making it possible to plot them
    assert len(set(model_lst)) == 1, "Can't plot if more than 1 unique family"
    
    ax1.plot(zz,[cop.parameters[0] for cop in cop_lst],label=model_lst[0])

    if model_lst[0] in pv.two_par: 
        ax2.plot(zz,[cop.parameters[1] for cop in cop_lst],label=model_lst[0])

    for i,ax in enumerate((ax1,ax2)):
        ax.set_xlabel('Z (micrometer)')
        ax.grid()
        ax.set_xlim(zz[0],zz[-1])
        ax.legend()
        ax.set_title(f'Parameter {i}')

    return

def get_L_and_phi(df_cleaned):
    df = df_cleaned.sort_values(['fibre_id', 'z']).copy()
    df['L'] = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2)
    df['phi'] = np.arctan2(df['dy'], df['dx'])
    return df

def reconstruct(df_clean,df_sim,zz_complete,n_fibers):
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
    sim_fibers[1:] += np.cumsum(np.tan(np.radians(df_sim[1:]))*z_scale,axis=0)


    # Stack the arrays to get data back in original shape
    stacked_sim_fibers = np.vstack(sim_fibers)
    
    # Create array in the format 0,1,2,...,n_fibers,0,1,2,...,n_fibers to repeat as many times as there are layers
    fibre_id_arr = np.tile(np.arange(n_fibers),len(zz_complete))
    # Create array in the format 0,0,0...,1,1,1,1..2,2,2,2 where each numbers repeats as many times as there are fibers
    zz_arr = np.repeat(zz_complete,n_fibers)

    df_columns = ['fibre_id','z_idx','x','y']

    sim_df = pd.DataFrame(columns=df_columns)
    sim_df['x'] = stacked_sim_fibers[:,0]
    sim_df['y'] = stacked_sim_fibers[:,1]
    sim_df['z_idx'] = zz_arr
    sim_df['fibre_id'] = fibre_id_arr
    sim_df['z'] = sim_df['z_idx'] * z_scale

    return sim_df
