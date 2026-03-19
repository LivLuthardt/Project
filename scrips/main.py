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
data_sim_arr = np.empty((len(cop_models),len(zz)+1,n_fibers,2))

cop_lst = [[] for i in range(len(cop_models))]

### Generate copulas for each z 
for z in zz:
    df_z = sort(df,z,par_1,par_2)
    for i,model in enumerate(cop_models):
        data_sim_arr[i,z], cop = bivariate_copula(df_z,n_fibers,model=model)
        cop_lst[i].append(cop)

### Plot covariance of Gaussian copulas
""" 
for cops in cop_lst:
    plot_cop_parameters(cops)
plt.subplot(1,2,1)
plt.plot(zz,cov_arr,label='Actual covariance')
plt.legend()

plt.show()
 """
plt.show()
# Plot og and synthetic data
# plot_og_data(par_1,par_2,mean_arr,df,[67])
plot_synthetic_data(par_1,par_2,mean_arr,df,data_sim_arr[1],[67])



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
