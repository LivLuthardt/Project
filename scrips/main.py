from data_clean import*
from ellipse import*
from tangent import*
from copula import*
from clustering import*
from plot import *
from layer_clustering import *


raw_df = pd.read_csv('raw_data.csv')
data_clean = data_cleaned(raw_df)
df = tangent_angles_central(data_clean)
fiber_sum,n_fibers = fiber_summary(df)

#ellipse 
xtiltAngles, ytiltAngles = [], [] #Init empty lists
first = True
for r in df.itertuples(index=True):
    x2 = (r[3], r[4], r[2]) #Current fiber point
    if first: tilt = (0, 0) #Can't compute tilt from a single point
    else:  tilt = eTiltAngles(x1, x2) #Pass the past and current points
    xtiltAngles.append(tilt[0])
    ytiltAngles.append(tilt[1])
    x1 = x2 #Set the current point to the past point
    first = False
df = df.assign(EllipseXTilt = xtiltAngles, EllipseYTilt = ytiltAngles) #Add the tilt angles as a df column
df = df.dropna(subset=['dx', 'dy', 'dz']) #Clean data
#Save 1D histograms
ax = df[["EllipseXTilt","angle_x_deg"]].plot.hist(bins=200, alpha=0.5, legend = True)
plt.savefig(fname="XTiltHist.png")
ax2 = df[["EllipseYTilt", "angle_y_deg"]].plot.hist(bins=200, alpha=0.5, legend = True)
plt.savefig(fname="YTiltHist.png")
#Save 2D hex plots
ax3 = df.plot.hexbin(x="EllipseXTilt", y="EllipseYTilt", gridsize=100, cmap="viridis", xlim = (-10, 10), ylim = (-10, 10))
plt.savefig(fname="EllipseTiltHex.png")
ax4 = df.plot.hexbin(x="angle_x_deg", y="angle_y_deg", gridsize=100, cmap="viridis", xlim = (-10, 10), ylim = (-10, 10))
plt.savefig(fname="FiniteTiltHex.png")
# Save and write standard deviations
fstd = (df[["angle_x_deg"]].std(), df[["angle_y_deg"]].std())
estd = (df[["EllipseXTilt"]].std(), df[["EllipseYTilt"]].std())
with open("Output.txt", "w") as text_file:
    text_file.write("Finite Difference Standard Deviations (x, y): %s" % str(fstd))
    text_file.write("Ellipse Method Standard Deviations (x, y): %s" % str(estd))

#copulas
zz = np.arange(1,128)
zz_complete = np.arange(129)

### Choose parameters to plot and predict with copula
par_1,par_2 = 'angle_x_deg','angle_y_deg'
df_grouped = df.groupby('z')

mean_arr = df_grouped.mean()[[par_1,par_2]].to_numpy()

### Magic
cov_series = df_grouped.apply(
    lambda group: group[par_1].corr(group[par_2]),
    include_groups=False)
cov_arr = cov_series.reindex(zz).to_numpy()

cop_models = [pv.gaussian,pv.student,pv.frank]

### Plot original data
# plot_og_data(par_1,par_2,mean_arr,df,[67])

# 129 is the amount of z values
# n_fibers is the amount of unique fibers
# 2 is the amount of parameters we can put in our copula
data_sim_arr = np.empty((len(cop_models),129,n_fibers,2))

# list to contain copulas 
# Generate a list with lists inside it
cop_lst = [[] for _ in range(len(cop_models))]

for z in zz:    #Iterate by layer
    df_z = sort(df,z,par_1,par_2) #Nested list of x, y tilts for the layer
    for i,model in enumerate(cop_models): #Iterate by copula model
        data_sim_arr[i,z], cop = bivariate_copula(df_z,n_fibers,model=model) #Construct a copula for layer tilts      
        if z > 1 and i==1:    
            #Get Pearson's r betweens layers
            xcor = pv.wdm(df_zp[:,0], df_z[:,0], 'cor')
            ycor = pv.wdm(df_zp[:,1], df_z[:,1], 'cor')
            #Set tilt angles using depth memory
            xtilt = data_sim_arr[i,z-1][:,0] * xcor + (data_sim_arr[i,z][:,0]) * (1-xcor**2) ** 0.5
            ytilt = data_sim_arr[i,z-1][:,1] * ycor + (data_sim_arr[i,z][:,1]) * (1-ycor**2) ** 0.5
            data_sim_arr[i,z] = np.concatenate([np.reshape(xtilt, (-1, 1)), np.reshape(ytilt, (-1, 1))], axis=1)
        elif z==1:
            data_sim_arr[i,0] = data_sim_arr[i,1] #Backwards fill data to initial layer
        cop_lst[i].append(cop)  #Add the copula to the list
    df_zp = df_z


for cops in cop_lst:
    print(f'Mean of {cops[0].family} AIC: {sum(cop.aic() for cop in cops)/len(cops):.2f}')

sim_df = reconstruct(data_clean,data_sim_arr[1],zz_complete,n_fibers)

# Plot synthetic fibers
fig = px.line_3d(sim_df[sim_df['fibre_id'] < 300],
                x="x", y="y", z="z",
                color="fibre_id",
                title=f'Synthetic Fibers')
fig.update_layout(
    scene=dict(aspectmode="manual",
            aspectratio=dict(x=15, y=7.5, z=1))
)
fig.show()

"""
### Plot copulas parameters
cop_fig, (ax5,ax6) = plt.subplots(1,2)

for cops in cop_lst:
    plot_cop_parameters(cops,ax5,ax6)

ax5.plot(zz,cov_arr,label='Actual correlation')

cop_fig.tight_layout()
cop_fig.savefig(fname='Copula_correlation',dpi=200)
print(f'Copula Correlation plot saved')
plt.close('all')

### Plot og and synthetic data
# plot_og_data(par_1,par_2,mean_arr,df,[67])
plot_synthetic_data(par_1,par_2,mean_arr,df,data_sim_arr[1],[30])

# ADD THE OTHER COLOUMNS TO SIMM_DF 

# apparently if we dont do this the objects cause everything to break (making floats)
sim_df[['x', 'y', 'z']] = sim_df[['x', 'y', 'z']].apply(pd.to_numeric)

sim_df = tangent_angles_central(sim_df)
sim_fiber_sum, n_sim_fibers = fiber_summary(sim_df)

# Save the new simulated date to file
sim_df[['fibre_id','x', 'y', 'z_idx']].to_csv('./sim_data.csv',sep=',',index=False,float_format="%.7f")

delaunay_triangulation(df)

#PCA method figure
pca, data_pca, coverage_lst = PCA_determination(fiber_sum)

# Number of pre-defined clusters and range for score plots
n = 5
n_clusters = range(2,16)

#K-means clustering
fiber_summary_k,_,_ = perform_kmeans_clustering(fiber_sum.copy(),n)
df_k = df.merge(fiber_summary_k[['fibre_id', 'cluster_id']], on='fibre_id')
# Make a plot of the error
#fig_k_error = sse_plot_k(fiber_sum)

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

# Make a plot of the error
#fig_gmm_error = aic_bic_plot_gmm(fiber_sum.copy())

fiber_summary_agg,_,_ = perform_agglomerative_clustering(fiber_sum,n)
df_agg = df.merge(fiber_summary_agg[['fibre_id', 'cluster_id']], on='fibre_id')

# Make 3D plots with clusters
# plot_fibers(df_k, 'K-means')
# plot_fibers(df_k_pca, 'K-means with PCA')
# plot_fibers(df_dbscan, 'DBSCAN')
# plot_fibers(df_hdbscan, 'HDBSCAN')
# plot_fibers(df_gmm, 'GMM')
# plot_fibers(df_agg, 'Agglomerative')

# Make score plot for all pre-defined cluster methods

# plot_score(fiber_sum, n_clusters)
# plot_sse_k(fiber_sum, n_clusters)
# plot_aic_bic_gmm(fiber_sum, n_clusters)

ks_x_list, ks_y_list = ks_by_z_lists(df)

print("KS X:", ks_x_list)
print("KS Y:", ks_y_list)

# neighbors(df) 
"""