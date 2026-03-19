import matplotlib.pyplot as plt

def plot_og_data(x1,x2,mean_arr,df,z_values=range(1,128)):
    for z in z_values:
        df_z = df[df['z'] == z]
        x1_df = df_z[[x1]]
        x2_df = df_z[[x2]]
        plt.subplot(1,3,1)
        plt.scatter(x1_df,x2_df)
        plt.scatter(mean_arr[z,0],mean_arr[z,1],color='k')
        plt.subplot(1,3,2)
        plt.hist(x1_df)
        plt.axvline(mean_arr[z,0], color='k', linestyle='dashed', linewidth=1)
        plt.subplot(1,3,3)
        plt.hist(x2_df)
        plt.axvline(mean_arr[z,0], color='k', linestyle='dashed', linewidth=1)
    plt.show()

def plot_synthetic_data(x1,x2,mean_arr,df,df_sim,z_values=range(1,128)):
    for z in z_values:
        plt.title(f'Scatterplot at z = {z}')
        df_z = df[df['z'] == z]
        x1_df = df_z[[x1]]
        x2_df = df_z[[x2]]
        plt.subplot(2,3,1)
        plt.scatter(x1_df,x2_df,label='Actual Data')
        plt.scatter(df_sim[:,0],df_sim[:,1],label='Synthetic')
        plt.scatter(mean_arr[z,0],mean_arr[z,1],color='k')
        plt.legend()

        plt.subplot(2,3,2)
        plt.hist(x1_df,label='Actual Data',bins=25)
        plt.axvline(mean_arr[z,0], color='k', linestyle='dashed', linewidth=1)
        plt.legend()

        plt.subplot(2,3,3)
        plt.hist(x2_df,label='Actual Data',bins=25)
        plt.axvline(mean_arr[z,1], color='k', linestyle='dashed', linewidth=1)
        plt.legend()

        plt.subplot(2,3,5)
        plt.hist(df_sim[z,:,0],label='Synthetic',bins=25)
        plt.legend()

        plt.subplot(2,3,6)
        plt.hist(df_sim[z,:,1],label='Synthetic',bins=25)
        plt.legend()

        plt.show()