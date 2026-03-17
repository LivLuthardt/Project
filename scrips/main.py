from data_clean import*
from ellipse import*
from tangent import*
from copula import*
from clustering import*
from plot import *

df = pd.read_csv('raw_data.csv')
data_clean = data_cleaned(df)
df = tangent_angles_central(data_clean)
fiber_sum,n_fibers = fiber_summary(df)

### Choose parameters to plot and predict with copula
par_1,par_2 = 'angle_x_deg','angle_y_deg'
mean_df = df.groupby('z').mean()
mean_arr = mean_df[[par_1,par_2]].to_numpy()

### Plot original data
plot_og_data(par_1,par_2,mean_arr,df,[67])

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
df = df.dropna(subset=['dx', 'dy', 'dz'])
ax = df[["EllipseXTilt","angle_x_deg"]].plot.hist(bins=200, alpha=0.5, legend = True)
plt.savefig(fname="XTiltHist.png")
ax2 = df[["EllipseYTilt", "angle_y_deg"]].plot.hist(bins=200, alpha=0.5, legend = True)
plt.savefig(fname="YTiltHist.png")
ax3 = df.plot.hexbin(x="EllipseXTilt", y="EllipseYTilt", gridsize=100, cmap="viridis", xlim = (-10, 10), ylim = (-10, 10))
plt.savefig(fname="EllipseTiltHex.png")
ax4 = df.plot.hexbin(x="angle_x_deg", y="angle_y_deg", gridsize=100, cmap="viridis", xlim = (-10, 10), ylim = (-10, 10))
plt.savefig(fname="FiniteTiltHex.png")
print(df)

# 129 is the amount of z values
# n_fibers is the amount of unique fibers
# 2 is the amount of parameters we can put in our copula
data_sim_lst = np.empty((129,n_fibers,2))

# list to contain copulas 
# TODO remove the 0 inside here once we have 129 istead of 128 datapoints
cop_lst = [0]

# Iterate [1,128] because for z = 0 certain parameters like dx and dy are undefined
# TODO once the dataframe is changed to account for z = 0 we can do range(129)
for row_n in range(1,128):
    data_filtered = sort(df,row_n,par_1,par_2)
    data_sim_lst[row_n], cop = bivariate_copula(data_filtered,len(data_filtered),family=pv.student)

    cop_lst.append(cop)

    if row_n % 5 == 0:
        continue
        print(f'Showing density plot for copula at z = {row_n}')
        cop_lst[row_n].plot('surface')
        # Scatter synthetic oberservation points
        plt.scatter(data_sim_lst[row_n,:,0],data_sim_lst[row_n,:,1])
        plt.title(f'Synthetic observations at z = {row_n}')
        plt.show()

    # TODO this is not correct but should remain until z = 0 is accounted for
    cop_lst[0] = cop_lst[1]

### Plot covariance of Gaussian copulas
# plot_cop_parameters(cop_lst)

# Number of pre-defined clusters
n = 5

fiber_summary_k = perform_kmeans_clustering(fiber_sum,n)
# 2. Merge cluster IDs back to the original points for 3D plotting
df_clustered_k = df.merge(fiber_sum[['fibre_id', 'cluster_id']], on='fibre_id')
fig_k = plot_k(df_clustered_k)
# make a plot of the error
fig_k_error = sse_plot_k(fiber_sum)

# DBSCAN clustering
fiber_summary_dbscan = perform_DBSCAN_clustering(fiber_sum)
df_clustered_dbscan = df.merge(fiber_sum[['fibre_id', 'cluster_id']], on='fibre_id')
fig_gmm = plot_DBSCAN(df_clustered_dbscan)

# HDBSCAN clustering
fiber_summary_hdbscan = perform_HDBSCAN_clustering(fiber_sum)
df_clustered_hdbscan = df.merge(fiber_sum[['fibre_id', 'cluster_id']], on='fibre_id')
fig_gmm = plot_HDBSCAN(df_clustered_hdbscan)

# GMM clustering
fiber_summary_gmm = perform_gmm_clustering(fiber_sum,n)
df_clustered_gmm = df.merge(fiber_sum[['fibre_id', 'cluster_id']], on='fibre_id')
fig_gmm = plot_gmm(df_clustered_gmm)
# Make a plot of the error
fig_gmm_error = aic_bic_plot_gmm(fiber_sum)

clustered, model = perform_agglomerative_clustering(df)
fig_agg = plot_agg(clustered)
