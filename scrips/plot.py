import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Ellipse
import numpy as np
import plotly.express as px
import pandas as pd
from clustering import perform_kmeans_clustering, perform_kmeans_clustering_with_pca, perform_gmm_clustering, perform_agglomerative_clustering
from copula import sort
import pyvinecopulib as pv

def plot_ellipse(df,z):
    """
    takes a dataframe and a z-level (0,128) and plots an ellipse intersection plot
    """
    df = df[df["z_idx"] == z]
    
    fig, ax = plt.subplots(figsize=(14*3,1*3))
    for i in range(len(df)):
        x = df.iloc[i]['x']
        y = df.iloc[i]['y']
        a = df.iloc[i]['a']
        b = df.iloc[i]['b']
        xytilt = df.iloc[i]['xytilt']

        ellipse = Ellipse(xy=(x, y), width=a*2, height=b*2, angle=np.degrees(xytilt), fill=False)
        ax.add_patch(ellipse)

    ax.set_xlim(-5, 1220) #change this to change the size of the plot, for the whole plot -5<x<1220, -170<y<5
    ax.set_ylim(-170, 5)

    ax.set_aspect('equal')
    plt.savefig(fname="EllipsePlot.png")

def plot_og_data(x1,x2,mean_arr,df,z_values=range(1,128)):
    for z in z_values:
        df_z = df[df['z_idx'] == z]
        x1_df = df_z[[x1]]
        x2_df = df_z[[x2]]
        plt.subplot(1,3,1)
        plt.scatter(x1_df,x2_df)
        plt.scatter(mean_arr[z,0],mean_arr[z,1],color='k')
        plt.subplot(1,3,2)
        plt.hist(x1_df,bins=150)
        plt.axvline(mean_arr[z,0], color='k', linestyle='dashed', linewidth=1)
        plt.subplot(1,3,3)
        plt.hist(x2_df,bins=150)
        plt.axvline(mean_arr[z,0], color='k', linestyle='dashed', linewidth=1)
    plt.show()

def plot_synthetic_data(x1,x2,mean_arr,std_arr,df,arr_sim,z_values=range(1,128)):
    for z in z_values:
        plt.close('all')

        # Take approriate data to plot
        df_z = df[df['z_idx'] == z]
        x1_df = df_z[[x1]].to_numpy()
        x2_df = df_z[[x2]].to_numpy()

        # Calulate mean and std from sim data
        sim_std_arr = np.std(arr_sim[z],axis=0)
        sim_mean_arr = np.mean(arr_sim[z])

        # Set the limits of the plot to be square based on the largest value in all sim and real data
        pltlims = max(np.abs(x1_df).max(),np.abs(x2_df).max(),np.abs(arr_sim).max())
        pltlims = (-pltlims*1.1,pltlims*1.1)

        plt.scatter(x1_df,x2_df,label='Actual Data',alpha=.3,edgecolors=None)
        plt.scatter(arr_sim[z,:,0],arr_sim[z,:,1],label='Synthetic',alpha=.3,edgecolors=None)
        plt.scatter(mean_arr[z,0],mean_arr[z,1],color='k')

        plt.xlim(pltlims), plt.ylim(pltlims)
        plt.legend(), plt.gca().set_aspect('equal'), plt.grid()
        plt.title(f'Scatterplot at z = {z} with Depth Memory')
        plt.xlabel(f'{x1}'),plt.ylabel(f'{x2}')
        plt.tight_layout()

        
        plt_str = (
            rf"Real: $\sigma_x = {std_arr[z,0]:.3f} \quad \sigma_y = {std_arr[z,1]:.3f}$" "\n"
            rf"Synthetic: $\sigma_x = {sim_std_arr[0]:.3f} \quad \sigma_y = {sim_std_arr[1]:.3f}$"
                    )
        
        text_box = AnchoredText(plt_str, loc='lower left', frameon=True, borderpad=0.0)

        text_box.patch.set_facecolor('white')
        text_box.patch.set_edgecolor('black')
        text_box.patch.set_alpha(1.0)
        # text_box.patch.set_boxstyle("square") # Keeps the internal padding around the text

        plt.gca().add_artist(text_box)

        plt.savefig(fname=f'Real_synthetic_scatterplot_z_{z}',dpi=200)
        print('Real and Synthetic scatterplot saved')
        plt.close('all')

        plt.subplot(2,2,1)
        plt.hist(x1_df,label='Actual Data',bins=150)
        plt.xlabel(f'{x1}')
        plt.axvline(mean_arr[z,0], color='k', linestyle='dashed', linewidth=1)
        plt.title('Actual Data')
        plt.xlim(pltlims)

        plt.subplot(2,2,2)
        plt.hist(x2_df,label='Actual Data',bins=150)
        plt.xlabel(f'{x2}')
        plt.title('Actual Data')
        plt.axvline(mean_arr[z,1], color='k', linestyle='dashed', linewidth=1)
        plt.xlim(pltlims)

        plt.subplot(2,2,3)
        plt.hist(arr_sim[z,:,0],label='Synthetic',bins=150)
        plt.xlabel(f'{x1}')
        plt.title('Synthetic')
        plt.axvline(arr_sim[z,:,0].mean(), color='k', linestyle='dashed', linewidth=1)
        plt.xlim(pltlims)

        plt.subplot(2,2,4)
        plt.hist(arr_sim[z,:,1],label='Synthetic',bins=150)
        plt.xlabel(f'{x2}')
        plt.title('Synthetic')
        plt.axvline(arr_sim[z,:,1].mean(), color='k', linestyle='dashed', linewidth=1)
        plt.xlim(pltlims)

        plt.tight_layout()
        plt.savefig(fname=f'Real_synthetic_histograms_z_{z}',dpi=200)

def single_fiber_plot(df,id):
    """
    plots the projection of a single fiber onto the xy plane to show the misalignment
    """
    df = df[df['fibre_id'] == id] #get a dataframe of one fiber

    #plot a 2D plot of the projection of the fiber onto the xy plane
    fig = px.line( 
        df, 
        x='x', 
        y='y', 
        markers=True,
        title=f"Fiber ID: {id}",
        line_shape='linear'
    )

    #fig.show()
    fig.write_image(f"Fiber_xy_proj_plot.png")

def sse_plot_kmeans_pca(df, n_components=3):
    """
    uses the clustered dataframe and the number of principal component
    to make a 3D-plot of the clustered fibers
    """
    sse_pca = []
    n_clusters_range = range(1, 11)

    #get the inertia (sum of squared distances of samples to their closest cluster center) 
    #of clustered dataframe for each number of clusters specified
    for k in n_clusters_range:
        _, inertia_pca, _, _ = perform_kmeans_clustering_with_pca(
            df,
            n_clusters=k,
            n_components=n_components
        )
        sse_pca.append(inertia_pca)

    #create a dataframe for plotting
    plot_df_pca = pd.DataFrame({
        'Number of Clusters': n_clusters_range,
        'SSE': sse_pca
    })

    #plot a 2D plot of the SSE per number of clusters
    fig = px.line(
        plot_df_pca,
        x='Number of Clusters',
        y='SSE',
        markers=True,
        title=f"SSE vs Number of Clusters (K-means with PCA, {n_components} PCs)"
    )
    #fig.show()

def plot_fibers(df,title):
    """
    uses a dataframe and title to make a 3D plot of all fibers in the dataframe
    """
    df = df[df['fibre_id'] < 100]#change/uncomment this if you want to reduce the number of fibers for faster computation
    #plot a 3D plot of the fibers per number of clusters
    fig = px.line_3d(
        df, 
        x='x', y='y', z='z',
        color='fibre_id',
        title=title
    )
    fig.update_layout(
    scene=dict(aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.4)) #change these values if you want to change the aspect ratio of the image
    )
    fig.show()

def plot_fibers_clustered(df,title):
    """
    uses a clustered dataframe and title to make a 3D plot of all clustered fibers in the dataframe
    """
    #df[df['fibre_id'] < 300] #change/uncomment this if you want to reduce the number of fibers for faster computation
    #plot a 3D plot of the clustered fibers per number of clusters
    fig = px.line_3d(
        df, 
        x='x', y='y', z='z', 
        color='cluster_id',
        line_group='fibre_id',
        title=title
    )
    fig.update_layout(
    scene=dict(aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1)) #change these values if you want to change the aspect ratio of the image
    )
    #fig.show()

def plot_score(df, n_clusters):
    """
    gets a dataframe and range of number of clusters to plot the Calinski–Harabasz index for k-means, GMM and agglomerative
    """
    score_list_k = []
    score_list_gmm = []
    score_list_agg = []
    #get CH index for all clustering methods
    for n in n_clusters:
        _,_, score_k = perform_kmeans_clustering(df,n)
        _,_,_, score_gmm = perform_gmm_clustering(df,n)
        _,_, score_agg = perform_agglomerative_clustering(df,n)
    
        score_list_k.append(score_k)
        score_list_gmm.append(score_gmm)
        score_list_agg.append(score_agg)


    #create a dataframe for plotting
    plot_df = pd.DataFrame({
        'Number of Clusters': list(n_clusters) * 3,
        'Calinski-Harabasz': score_list_k + score_list_gmm + score_list_agg,
        'Method': ['K-means'] * len(n_clusters) + ['GMM'] * len(n_clusters) + ['Agglomerative'] * len(n_clusters)
    })

    #plot a 2D plot of the CH index per number of clusters
    fig = px.line(
        plot_df,
        x='Number of Clusters',
        y='Calinski-Harabasz',
        color='Method',
        markers=True,
        title="Calinski-Harabasz score vs Number of Clusters for Clustering Methods"
    )

    #fig.show()
    fig.write_image(f"CD_score_plot.png")
    print(f'Plot CD finished')

def plot_sse_k(df, n_clusters):
    """
    gets a dataframe and range of number of clusters to plot the SSE for k-means
    """
    sse = []

    #get the inertia (sum of squared distances of samples to their closest cluster center) 
    #of clustered dataframe for each number of clusters specified
    for k in n_clusters:
        _ , inertia, _ = perform_kmeans_clustering(df,n_clusters=k)
        sse.append(inertia)

    #create a dataframe for plotting
    plot_df = pd.DataFrame({
        'Number of Clusters': n_clusters,
        'SSE': sse
    })

    #plot a 2D plot of the SSE per number of clusters
    fig = px.line(
        plot_df, 
        x='Number of Clusters', 
        y='SSE', 
        markers=True,
        title="SSE vs Number of Clusters for K-means"
    )

    #fig.show()
    fig.write_image(f"SSE_k_means.png")
    print(f'Plot SSE K-means finished')

def plot_aic_bic_gmm(df, n_clusters):
    """
    gets a dataframe and range of number of clusters to plot the AIC and BIC for gmm clustering
    """
    aic_vals = []
    bic_vals = []
    # get the AIC and BIC
    for k in n_clusters:
        _ , aic , bic, _ = perform_gmm_clustering(df,n_clusters=k)
        aic_vals.append(aic)
        bic_vals.append(bic)

    #create a dataframe for plotting
    plot_df = pd.DataFrame({
        'Number of Clusters': list(n_clusters) * 2,
        'Criterion': ['AIC'] * len(n_clusters) + ['BIC'] * len(n_clusters),
        'Value': aic_vals + bic_vals
    })

    #plot a 2D plot of the AIC and BIC per number of clusters
    fig = px.line(
        plot_df, 
        x='Number of Clusters', 
        y='Value', 
        color='Criterion',
        markers=True,
        title="AIC and BIC vs Number of Clusters for GMM",
        labels={'Value': 'Criterion Value'}
    )

    #fig.show()
    fig.write_image(f"AIC_BIC_GMM.png")
    print(f'Plot AIC BIC GMM finished')

def One_D_ellipse_tilt_hist(df):
    plt.figure()
    ax = df[["EllipseXTilt","angle_x_deg"]].plot.hist(bins=200, alpha=0.5, legend = True)
    ax.set_title('Fiber x-tilt Histogram')
    ax.set_xlabel('Fiber angle x-tilt (°)')
    ax.set_ylabel('Frequency')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Ellipse method", "Finite difference method"])
    plt.savefig(fname="XTiltHist.png")
    plt.close('all')
    plt.figure()
    ax = plt.gca()
    ax = df[["EllipseYTilt", "angle_y_deg"]].plot.hist(bins=200, alpha=0.5, legend = True)
    ax.set_title('Fiber y-tilt Histogram')
    ax.set_xlabel('Fiber angle y-tilt (°)')
    ax.set_ylabel('Frequency')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Ellipse method", "Finite difference method"])
    plt.savefig(fname="YTiltHist.png")
    plt.close('all')

def Two_D_hex_plot(df):
    ax3 = df.plot.hexbin(x="EllipseXTilt", y="EllipseYTilt", gridsize=100, cmap="viridis", xlim = (-10, 10), ylim = (-10, 10))
    plt.savefig(fname="EllipseTiltHex.png")
    ax4 = df.plot.hexbin(x="angle_x_deg", y="angle_y_deg", gridsize=100, cmap="viridis", xlim = (-10, 10), ylim = (-10, 10))
    plt.savefig(fname="FiniteTiltHex.png")

def plot_alpha_z(data_raw,data_sim_arr,cop_models):
    """ 
    Take raw data and simulated data and plot the absolute mean of
    fiber angle projected on xy-plane (alpha in literature) 
    """
    theta_z_sim = np.degrees(np.atan2(np.radians(data_sim_arr[:,:,:,0]),np.radians(data_sim_arr[:,:,:,1])))
    theta_z_sim_mean = np.abs(np.mean(theta_z_sim,axis=2))
    
    theta_z_raw_mean = np.empty(129)

    for z in range(129):
        data_z = sort(data_raw,z)
        theta_z_raw = np.degrees(np.atan2(np.radians(data_z[:,0]),np.radians(data_z[:,1])))
        theta_z_raw_mean[z] = np.abs(np.mean(theta_z_raw))

    plt.close('all')

    plt.plot(np.arange(129)*500/128,theta_z_sim_mean[1],label=f'Student Copula with Depth Memory')
    plt.plot(np.arange(129)*500/128,theta_z_sim_mean[2],label=f'Student Copula without Depth Memory')

    plt.plot(np.arange(129)*500/128,theta_z_raw_mean,label='Raw fibers')
    plt.legend()
    plt.xlabel(rf'''z [$\mu m$]'''), plt.ylabel(rf'''$\theta_z$ [deg]''')
    plt.grid()
    plt.savefig(fname='mean_alpha_z',dpi=200)
    plt.close('all')

def plot_theta_z(data_raw,data_sim_dm,data_sim):
    """ 
    Take raw data and simulated data and plot the absolute mean of
    fiber angle projected on xy-plane (alpha in literature) 
    """

    plt.close('all')
    z_scale = 500/128


    for df in (data_raw,data_sim_dm,data_sim):
        df['r'] = np.hypot(df['x'],df['y'])
        df['theta_z'] = np.abs(np.degrees(np.arctan(z_scale/df['r'])))

    raw_mean_theta_z = data_raw.groupby('z')['theta_z'].mean()
    sim_dm_mean_theta_z = data_sim_dm.groupby('z')['theta_z'].mean()
    sim_mean_theta_z = data_sim.groupby('z')['theta_z'].mean()

    # plt.plot(np.arange(129)*z_scale,df['theta_z'])
    plt.plot(raw_mean_theta_z,label='Raw')
    plt.plot(sim_dm_mean_theta_z,label='Depth mem')
    plt.plot(sim_mean_theta_z,label='No depth')
    
    plt.ylim(.4,.7)
    plt.legend()
    plt.xlabel(rf'''z [$\mu m$]'''), plt.ylabel(rf'''$\theta_z$ [deg]''')
    plt.grid()
    plt.savefig(fname='mean_theta_z',dpi=200)
    plt.close('all')

def plot_correlation(zz,x1,x2,dfs,labels):
    """ 
    Plot copula parameters as a function of Z
    Put in ax1 and ax2 to be able to plot parameters for different copula collections
    """

    fig,ax = plt.subplots() 

    # Calculate the correlation for each dataframe
    # Given that we have 3 dataframes (raw, with dm, without dm) we need to loop
    for i,df in enumerate(dfs):
        df_grouped = df.groupby('z_idx')
        corr_series = df_grouped.apply(
        lambda group: group[x1].corr(group[x2]),
        include_groups=False)

        corr_arr = corr_series.reindex(zz).to_numpy()
        ax.plot(zz*(500/128),corr_arr,label=labels[i])

    ax.set_ylabel('Correlation')
    ax.set_xlabel(rf'z ($\mu m$)')
    ax.grid()
    # ax.set_xlim(zz[0],zz[-1])
    ax.legend()

    fig.tight_layout()
    fig.savefig(fname='Copula_correlation',dpi=200)
    plt.close('all')
