from data_clean import*
from ellipse import*
from tangent import*
from copula import*
from clustering import*
from plot import *
import matplotlib.pyplot as plt

raw_df = pd.read_csv('raw_data.csv')
data_clean = data_cleaned(raw_df)
df = tangent_angles_central(data_clean)
fiber_sum,n_fibers = fiber_summary(df)

#plot_fibers(df,'Original Fibers')
#-------------------------------------------------------------Ellipse-------------------------------------------------------------

xtiltAngles, ytiltAngles, xytiltAngles, alist, blist = getEllipseValues(data_clean)
df = df.assign(EllipseXTilt = xtiltAngles, EllipseYTilt = ytiltAngles, xytilt = xytiltAngles, a = alist, b = blist) #Add the tilt angles as a df column
df = df.dropna(subset=['dx', 'dy', 'dz']) #Clean data

#Ellipse plot
plot_ellipse(df,1)

#Single fiber projection plot
single_fiber_plot(df,5)

#Make 1D histograms of ellipsetilt vs angle 
One_D_ellipse_tilt_hist(df)

#Make 2D hex plot of ellipsetilt
Two_D_hex_plot(df)

#Save and write standard deviations
fstd = (df[["angle_x_deg"]].std(), df[["angle_y_deg"]].std())
estd = (df[["EllipseXTilt"]].std(), df[["EllipseYTilt"]].std())

fmean_x = df["angle_x_deg"].mean()
fmean_y = df["angle_y_deg"].mean()

emean_x = df["EllipseXTilt"].mean()
emean_y = df["EllipseYTilt"].mean()
with open("Output.txt", "w") as text_file:
    text_file.write("Finite Difference Standard Deviations (x, y): %s" % str(fstd))
    text_file.write("Ellipse Method Standard Deviations (x, y): %s" % str(estd))

    text_file.write(f"Finite Difference Mean (x, y): {fmean_x}, {fmean_y}\n")
    text_file.write(f"Ellipse Method Mean (x, y): {emean_x}, {emean_y}\n")

#-------------------------------------------------------------Copulas-------------------------------------------------------------

#copulas
zz = np.arange(1,128)
zz_complete = np.arange(129)

### Choose parameters to plot and predict with copula
par_1,par_2 = 'angle_x_deg','angle_y_deg'
df_grouped = df.groupby('z_idx')

mean_arr = df_grouped.mean()[[par_1,par_2]].to_numpy()
std_arr = df_grouped.std()[[par_1,par_2]].to_numpy()

### Magic
cov_series = df_grouped.apply(
    lambda group: group[par_1].corr(group[par_2]),
    include_groups=False)
cov_arr = cov_series.reindex(zz).to_numpy()

cop_models = [pv.gaussian,pv.student,pv.student,pv.frank]

### Plot original data
# plot_og_data(par_1,par_2,mean_arr,df,[67])

# 129 is the amount of z values
# n_fibers is the amount of unique fibers
# 2 is the amount of parameters we can put in our copula
data_sim_arr = np.empty((len(cop_models),129,n_fibers,2))

# list to contain copulas 
# Generate a list with lists inside it
cop_lst = [[] for _ in range(len(cop_models))]
u_lst = [[] for _ in range(len(zz)+1)]

#Prepopulate u for student model
for z in zz:
    df_z = sort(df, z, par_1, par_2)
    u_lst[z], cop = bivar_cop_u(df_z, n_fibers) #u corresponds to the appropriate z layer
    cop_lst[1].append(cop)

for z in zz:    #Iterate by layer
    df_z = sort(df,z,par_1,par_2) #Nested list of x, y tilts for the layer
    for i,model in enumerate(cop_models): #Iterate by copula model
        if i != 1 or (i == 1 and z == 1):
            data_sim_arr[i,z], cop = bivariate_copula(df_z,n_fibers,model=model) #Construct a copula for layer tilts      
        elif z > 1 and i==1:    
            cor = np.array([pv.wdm(df_zp[:,0], df_z[:,0], 'rho'), pv.wdm(df_zp[:,1], df_z[:,1], 'rho')]) #Get Spearman's rho betweens layers
            rho_g = 2 * np.sin(np.pi / 6 * cor) #Adapt rho for the z-domain
            data_sim_arr[i,z], u_lst[z] = depth_mem(df_z, (u_lst[z-1], u_lst[z]), rho_g) #Set tilt angles using depth memory
        if z==1:
            data_sim_arr[i,0] = data_sim_arr[i,1] #Backwards fill data to initial layer
        if z==127:
            data_sim_arr[i,128] = data_sim_arr[i,127] #Forward fill data to last layer
        if i != 1:
            cop_lst[i].append(cop)  #Add the copula to the list
    df_zp = df_z #Update the variable for the previous layer tilt

# for cops in cop_lst:
#     print(f'Mean of {cops[0].family} AIC with depth memory: {sum(cop.aic() for cop in cops)/len(cops):.2f}')

sim_df_dm = reconstruct(data_clean,data_sim_arr[1],zz_complete,n_fibers,par_1,par_2)
sim_df = reconstruct(data_clean,data_sim_arr[2],zz_complete,n_fibers,par_1,par_2)

# Copula contour plots
# plt.close('all')
# cop_lst[2][30].plot(type='contour',margin_type='unif')
# cop_lst[2][30].plot(type='contour',margin_type='norm')

# Plot synthetic fibers
#plot_fibers(sim_df_dm,'Synthetic Fibers with Depth Memory')
#plot_fibers(sim_df,'Synthetic Fibers without Depth Memory')


### Plot og and synthetic data
# plot_og_data(par_1,par_2,mean_arr,df,[67])
plot_synthetic_data(par_1,par_2,mean_arr,std_arr,df,data_sim_arr[1],[30,60])

plot_alpha_z(df,data_sim_arr,cop_models)
plot_theta_z(df,sim_df_dm)
plot_theta_x(df,sim_df_dm)
plot_theta_y(df,sim_df_dm)
plot_correlation(zz,par_1,par_2,(df,sim_df_dm,sim_df),
                 labels=['Raw Data',
                         'Simulated with Depth Memory',
                         'Simulated w/o Depth Memory '])

chi_squared_2d(df,data_sim_arr,cop_models)
chi_squared_1d(par_1,par_2,df,data_sim_arr,cop_models,zz)

# ADD THE OTHER COLOUMNS TO SIMM_DF 

# apparently if we dont do this the objects cause everything to break (making floats)
sim_df_dm[['x', 'y', 'z']] = sim_df_dm[['x', 'y', 'z']].apply(pd.to_numeric)

sim_df_dm = tangent_angles_central(sim_df_dm)
sim_fiber_sum, n_sim_fibers = fiber_summary(sim_df_dm)

# Save the new simulated date to file
sim_df_dm[['fibre_id','x', 'y', 'z_idx']].to_csv('./sim_data.csv',sep=',',index=False,float_format="%.7f")

#-------------------------------------------------------------Global clustering (not used anymore)-------------------------------------------------------------

#PCA method figure
pca, data_pca, coverage_lst = PCA_determination(fiber_sum)

# Number of pre-defined clusters and range for score plots
n = 5
n_clusters = range(2,16)

#K-means clustering
fiber_summary_k,_,_ = perform_kmeans_clustering(fiber_sum.copy(),n)
df_k = df.merge(fiber_summary_k[['fibre_id', 'cluster_id']], on='fibre_id')

# K-means clustering with PCA
fiber_summary_k_pca,_,_,_ = perform_kmeans_clustering_with_pca(
    fiber_sum, n_clusters=n, n_components=3)
df_k_pca = df.merge(fiber_summary_k_pca[['fibre_id', 'cluster_id']], on='fibre_id')

# DBSCAN clustering
fiber_summary_dbscan,_ = perform_DBSCAN_clustering(fiber_sum.copy())
df_dbscan = df.merge(fiber_summary_dbscan[['fibre_id', 'cluster_id']], on='fibre_id')

# HDBSCAN clustering
fiber_summary_hdbscan,_ = perform_HDBSCAN_clustering(fiber_sum.copy())
df_hdbscan = df.merge(fiber_summary_hdbscan[['fibre_id', 'cluster_id']], on='fibre_id')

# GMM clustering
fiber_summary_gmm,_,_,_ = perform_gmm_clustering(fiber_sum.copy(),n)
df_gmm = df.merge(fiber_summary_gmm[['fibre_id', 'cluster_id']], on='fibre_id')

fiber_summary_agg,_,_ = perform_agglomerative_clustering(fiber_sum,n)
df_agg = df.merge(fiber_summary_agg[['fibre_id', 'cluster_id']], on='fibre_id')

# Make 3D plots with clusters
# plot_fibers_clustered(df_k, 'K-means')
# plot_fibers_clustered(df_k_pca, 'K-means with PCA')
# plot_fibers_clustered(df_dbscan, 'DBSCAN')
# plot_fibers_clustered(df_hdbscan, 'HDBSCAN')
# plot_fibers_clustered(df_gmm, 'GMM')
# plot_fibers_clustered(df_agg, 'Agglomerative')

# Make CH score plot for all pre-defined cluster methods, as well as aic and bic for gmm and sse for k-means 

# plot_score(fiber_sum, n_clusters)
# plot_sse_k(fiber_sum, n_clusters)
# plot_aic_bic_gmm(fiber_sum, n_clusters)
#fig_gmm_error = aic_bic_plot_gmm(fiber_sum.copy())
#fig_k_error = sse_plot_k(fiber_sum)

# neighbors(df) 

ks_x_cd, ks_y_cd = ks_global(df)  # original (finite diff vs ellipse)

# Central difference vs ellipse
ks_x_cd_stat = ks_2samp(df["angle_x_deg"].dropna(), df["EllipseXTilt"].dropna())
ks_y_cd_stat = ks_2samp(df["angle_y_deg"].dropna(), df["EllipseYTilt"].dropna())

# Synthetic vs ellipse
ks_x_syn = ks_2samp(sim_df_dm["angle_x_deg"].dropna(), df["angle_x_deg"].dropna())
ks_y_syn = ks_2samp(sim_df_dm["angle_y_deg"].dropna(), df["angle_y_deg"].dropna())

with open("Output.txt", "a") as text_file:
    text_file.write(f"\nKS Test Central Diff (X): statistic={ks_x_cd.statistic:.4f}, p={ks_x_cd.pvalue:.4e}\n")
    text_file.write(f"KS Test Central Diff (Y): statistic={ks_y_cd.statistic:.4f}, p={ks_y_cd.pvalue:.4e}\n")
    text_file.write(f"\nKS Test Synthetic (X): statistic={ks_x_syn.statistic:.4f}, p={ks_x_syn.pvalue:.4e}\n")
    text_file.write(f"KS Test Synthetic (Y): statistic={ks_y_syn.statistic:.4f}, p={ks_y_syn.pvalue:.4e}\n")