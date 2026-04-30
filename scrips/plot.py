import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np

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
        plt.title(f'Scatterplot at z = {z}')
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
        plt.close()

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