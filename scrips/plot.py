import matplotlib.pyplot as plt

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

def plot_synthetic_data(x1,x2,mean_arr,df,df_sim,z_values=range(1,128)):
    for z in z_values:
        df_z = df[df['z_idx'] == z]
        x1_df = df_z[[x1]]
        x2_df = df_z[[x2]]
        plt.scatter(x1_df,x2_df,label='Actual Data',alpha=.3,edgecolors=None)
        plt.scatter(df_sim[z,:,0],df_sim[z,:,1],label='Synthetic',alpha=.3,edgecolors=None)
        plt.scatter(mean_arr[z,0],mean_arr[z,1],color='k')
        plt.legend(), plt.gca().set_aspect('equal'), plt.grid()
        plt.title(f'Scatterplot at z = {z}')
        plt.xlabel(f'{x1}'),plt.ylabel(f'{x2}')
        plt.tight_layout()
        plt.savefig(fname=f'Real_synthetic_scatterplot_z_{z}',dpi=200)
        print('Real and Synthetic scatterplot saved')
        plt.close()

        plt.subplot(2,2,1)
        plt.hist(x1_df,label='Actual Data',bins=150)
        plt.xlabel(f'{x1}')
        plt.axvline(mean_arr[z,0], color='k', linestyle='dashed', linewidth=1)
        plt.title('Actual Data')
        plt.xlim(-20,20)

        plt.subplot(2,2,2)
        plt.hist(x2_df,label='Actual Data',bins=150)
        plt.xlabel(f'{x2}')
        plt.title('Actual Data')
        plt.axvline(mean_arr[z,1], color='k', linestyle='dashed', linewidth=1)
        plt.xlim(-20,20)

        plt.subplot(2,2,3)
        plt.hist(df_sim[z,:,0],label='Synthetic',bins=150)
        plt.xlabel(f'{x1}')
        plt.title('Synthetic')
        plt.axvline(df_sim[z,:,0].mean(), color='k', linestyle='dashed', linewidth=1)
        plt.xlim(-20,20)

        plt.subplot(2,2,4)
        plt.hist(df_sim[z,:,1],label='Synthetic',bins=150)
        plt.xlabel(f'{x2}')
        plt.title('Synthetic')
        plt.axvline(df_sim[z,:,1].mean(), color='k', linestyle='dashed', linewidth=1)
        plt.xlim(-20,20)

        plt.tight_layout()
        plt.savefig(fname=f'Real_synthetic_histograms_z_{z}',dpi=200)