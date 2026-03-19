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
    # Return tilt angles for a given z-value, first indexing by layer and then by tilt outputs
    return data[data['z'] == n][[x1,x2]].to_numpy()

def bivariate_copula(data,n,model=None): #n is number of fibers in a layer
    u = pv.to_pseudo_obs(data)
    # pv.pairs_copula_data(u, scatter_size=0.5)

    # If no family is specified, this means that cop.select will run to choose a family
    # You should avoid running without specifying a family because it takes ages to run
    if model:
        cop = pv.Bicop(model)
        cop.fit(data=u)
    else:
        cop = pv.Bicop()
        cop.select(data=u)


    u_sim = cop.simulate(n)
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

def gen_copula(df,x1,x2):
    pass
    return data_sim,cop

def plot_cop_parameters(cop_lst):
    zz = np.arange(len(cop_lst))
    model_lst = [cop.family for cop in cop_lst]
    model_set = set(model_lst)

    assert len(model_set) == 1, "Can't plot if more than 1 unique family"

    if model_lst[0] in pv.one_par:
        plt.subplot(1,2,1)
        plt.plot(zz,[cop.parameters[0] for cop in cop_lst],label=model_lst[0])
    
    if model_lst[0] in pv.two_par:
        # First subplot
        plt.subplot(1,2,1)
        plt.plot(zz,[cop.parameters[0] for cop in cop_lst],label=model_lst[0])

        # Second subplot
        plt.subplot(1,2,2)
        plt.plot(zz,[cop.parameters[1] for cop in cop_lst],label=model_lst[0])

    for i in (1,2):
        plt.subplot(1,2,i)
        plt.xlabel('Z (micrometer)')
        plt.grid()
        plt.xlim(0,128)
        plt.legend()
        plt.title(f'Parameter {i}')

    return

def get_L_and_phi(df_cleaned):
    df = df_cleaned.sort_values(['fibre_id', 'z']).copy()
    df['L'] = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2)
    df['phi'] = np.arctan2(df['dy'], df['dx'])
    return df

def coordinates(arr, df_clean):
    dz = 1
    df_synthetic = pd.DataFrame(arr, columns=['angle_x_deg', 'angle_y_deg'])
    df_synthetic[['x', 'y', 'z']] = df_clean.loc[[0], ['x', 'y', 'z']]
    for i in range(1,129):
        df_synthetic['z'] = df_synthetic[[i]]['z'] + dz
    # df_synthetic[]

    return df_synthetic

def reconstruct(df_clean,df_sim,zz,n_fibers):
    sim_fibers = np.zeros((len(zz),n_fibers,2))

    df_0 = sort(df_clean,0,'x','y')

    sim_fibers[:,:,:] += df_0

    for i in range(len(zz)):
        sim_fibers[i:,:,:] += np.arctan(np.degrees(df_sim[i,:,:]))

    return sim_fibers
