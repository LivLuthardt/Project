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

zz = range(1,128)

### Choose parameters to plot and predict with copula
par_1,par_2 = 'angle_x_deg','angle_y_deg'
df_grouped = df.groupby('z')

mean_arr = df_grouped.mean()[[par_1,par_2]].to_numpy()

### Magic
cov_series = df_grouped.apply(lambda group: group[par_1].corr(group[par_2]))
cov_arr = cov_series.reindex(zz).to_numpy()

cop_models = [pv.gaussian,pv.student,pv.clayton]

### Plot original data
# plot_og_data(par_1,par_2,mean_arr,df,[67])

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
# print(df)
fstd = (df[["angle_x_deg"]].std(), df[["angle_y_deg"]].std())
estd = (df[["EllipseXTilt"]].std(), df[["EllipseYTilt"]].std())
with open("Output.txt", "w") as text_file:
    text_file.write("Finite Difference Standard Deviations (x, y): %s" % str(fstd))
    text_file.write("Ellipse Method Standard Deviations (x, y): %s" % str(estd))
# 129 is the amount of z values
# n_fibers is the amount of unique fibers
# 2 is the amount of parameters we can put in our copula
data_sim_arr = np.empty((len(cop_models),129,n_fibers,2))

# list to contain copulas 
# TODO remove the 0 inside here once we have 129 istead of 128 datapoints
cop_lst = [[] for i in range(len(cop_models))]

# Iterate [1,128] because for z = 0 certain parameters like dx and dy are undefined
# TODO once the dataframe is changed to account for z = 0 we can do range(129)
for z in zz:
    df_z = sort(df,z,par_1,par_2)
    for i,model in enumerate(cop_models):
        data_sim_arr[i,z], cop = bivariate_copula(df_z,n_fibers,model=model)
        cop_lst[i].append(cop)

    if z % 5 == 0:
        continue
        print(f'Showing density plot for copula at z = {z}')
        cop_lst[z].plot('surface')
        # Scatter synthetic oberservation points
        plt.scatter(data_sim_arr[z,:,0],data_sim_arr[z,:,1])
        plt.title(f'Synthetic observations at z = {z}')
        plt.show()

### Plot covariance of Gaussian copulas
""" 
for cops in cop_lst:
    plot_cop_parameters(cops)
plt.subplot(1,2,1)
plt.plot(zz,cov_arr,label='Actual covariance')
plt.legend()

plt.show()
"""
#PCA method figure
pca, data_pca, coverage_lst = PCA_determination(fiber_sum)

# Number of pre-defined clusters and range for silhouette plots
n = 5
n_clusters = range(2,16)

#K-means clustering
fiber_summary_k = perform_kmeans_clustering(fiber_sum,n)
# 2. Merge cluster IDs back to the original points for 3D plotting
df_clustered_k = df.merge(fiber_sum[['fibre_id', 'cluster_id']], on='fibre_id')
# Make a plot of the error
#fig_k_error = sse_plot_k(fiber_sum)


# K-means clustering with PCA
fiber_summary_k_pca, inertia_k_pca, score_k_pca, explained_var_pca = perform_kmeans_clustering_with_pca(
    fiber_sum, n_clusters=n, n_components=3)
df_clustered_k_pca = df.merge(fiber_summary_k_pca[['fibre_id', 'cluster_id']], on='fibre_id')

# DBSCAN clustering
fiber_summary_dbscan = perform_DBSCAN_clustering(fiber_sum)
df_clustered_dbscan = df.merge(fiber_sum[['fibre_id', 'cluster_id']], on='fibre_id')

# HDBSCAN clustering
fiber_summary_hdbscan = perform_HDBSCAN_clustering(fiber_sum)
df_clustered_hdbscan = df.merge(fiber_sum[['fibre_id', 'cluster_id']], on='fibre_id')

# GMM clustering
fiber_summary_gmm = perform_gmm_clustering(fiber_sum,n)
df_clustered_gmm = df.merge(fiber_sum[['fibre_id', 'cluster_id']], on='fibre_id')
# Make a plot of the error
#fig_gmm_error = aic_bic_plot_gmm(fiber_sum)

#df_clustered_agg, model, score = perform_agglomerative_clustering(fiber_sum,n)

'''
# Make 3D plots with clusters
plot_fibers(df_clustered_k, 'K-means')
plot_fibers(df_clustered_k_pca, 'K-means with PCA')
plot_fibers(df_clustered_dbscan, 'DBSCAN')
plot_fibers(df_clustered_hdbscan, 'HDBSCAN')
plot_fibers(df_clustered_gmm, 'GMM')
#plot_fibers(df_clustered_agg, 'agglomerative')
'''
"""

# Make silhouette plot for all pre-defined cluster methods
score_k = []
score_gmm = []
score_agg = []


for n in n_clusters:
    fiber_summary_k = perform_kmeans_clustering(fiber_sum,n)
    df_clustered_k = df.merge(fiber_sum[['fibre_id', 'cluster_id']], on='fibre_id')

    fiber_summary_gmm = perform_gmm_clustering(fiber_sum,n)
    df_clustered_gmm = df.merge(fiber_sum[['fibre_id', 'cluster_id']], on='fibre_id')

    #df_clustered_agg, model = perform_agglomerative_clustering(fiber_sum,n)

    score_k.append(fiber_summary_k[2])
    score_gmm.append(fiber_summary_gmm[3])
    #score_agg.append(df_clustered_agg[2])

plot_silhouette(score_k, n_clusters, 'K-means')
plot_silhouette(score_k, n_clusters, 'GMM')
plot_silhouette(score_k, n_clusters, 'Agglomerative')

#I will try to fix this next shesh

"""



