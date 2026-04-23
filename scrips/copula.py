import pyvinecopulib as pv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    """
    z_scale = 500 / 128

    # Create empty array to accomodate new fivers
    sim_fibers = np.zeros((len(zz_complete),n_fibers,2))

    # Take base layer of measured fibers
    df_0 = sort(df_clean,0,'x','y')

    # Add starting points to each corresponding fiber
    sim_fibers[:,:,:] += df_0

    # Cummulatively sum the dx and dy values by converting angles to distance
    for i in range(len(zz_complete) - 1):
        sim_fibers[i+1:,:,:] += np.tan(np.radians(df_sim[i,:,:])) * z_scale

    return sim_fibers
