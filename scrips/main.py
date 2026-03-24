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
cop_lst = [[] for i in range(len(cop_models))]
aic_gaussian = []
aic_student = []
aic_frank = []
for z in zz:
    df_z = sort(df,z,par_1,par_2)
    for i,model in enumerate(cop_models):
        data_sim_arr[i,z], cop, aic = bivariate_copula(df_z,n_fibers,model=model)
        cop_lst[i].append(cop)
        if f"{model}" == 'BicopFamily.gaussian':
            aic_gaussian.append(aic)
        elif f"{model}" == 'BicopFamily.student':
            aic_student.append(aic)
        elif f"{model}" == 'BicopFamily.frank':
            aic_frank.append(aic)
        else:
            print('aaaa')
    for i in range(len(cop_models)):
        data_sim_arr[i,0] = data_sim_arr[i,1]

gaussian_mean = np.mean(aic_gaussian)
student_mean = np.mean(aic_student)
frank_mean = np.mean(aic_frank)
# print(gaussian_mean)
# print(student_mean)
# print(frank_mean)

sim_fibers = reconstruct(data_clean,data_sim_arr[0],zz,n_fibers)

df_columns = ['fibre_id','z','x','y']

sim_df = pd.DataFrame(columns=df_columns)

for fibre_id in range(n_fibers):
    new_rows = np.empty((129,4),dtype=object)

    new_rows[:,-2:] = np.round(sim_arr[:,fibre_id,:],4)
    new_rows[:,0] = fibre_id
    new_rows[:,1] = zz_complete

    new_df = pd.DataFrame(new_rows, columns=df_columns)

    sim_df = pd.concat([sim_df, new_df],ignore_index=True)

# Plot synthetic fibers

# fig = px.line_3d(sim_fibers_df,
#                 x="x", y="y", z="z",
#                 color="fibre_id",
#                 title='Synthetic Fibers')
# fig.update_layout(
#     scene=dict(aspectmode="manual",
#             aspectratio=dict(x=15, y=7.5, z=1))
# )
# fig.show()


### Plot copulas parameters
# cop_fig, (ax5,ax6) = plt.subplots(1,2)

# for cops in cop_lst:
#     plot_cop_parameters(cops,ax5,ax6)

# ax5.plot(zz,cov_arr,label='Actual correlation')

# cop_fig.tight_layout()
# cop_fig.savefig(fname='Copula_correlation',dpi=200)
# print(f'Copula Correlation plot saved')
# plt.close('all')

### Plot og and synthetic data
# plot_og_data(par_1,par_2,mean_arr,df,[67])
# plot_synthetic_data(par_1,par_2,mean_arr,df,data_sim_arr[1],[30])
"""

#PCA method figure
pca, data_pca, coverage_lst = PCA_determination(fiber_sum)
"""
# Number of pre-defined clusters and range for score plots
n = 5
n_clusters = range(2,16)

#K-means clustering
fiber_summary_k, inertia_k, score_k = perform_kmeans_clustering(fiber_sum.copy(),n)
df_k = df.merge(fiber_summary_k[['fibre_id', 'cluster_id']], on='fibre_id')
# Make a plot of the error
#fig_k_error = sse_plot_k(fiber_sum)

# K-means clustering with PCA
fiber_summary_k_pca, inertia_k_pca, score_k_pca, explained_var_pca = perform_kmeans_clustering_with_pca(
    fiber_sum, n_clusters=n, n_components=3)
df_k_pca = df.merge(fiber_summary_k_pca[['fibre_id', 'cluster_id']], on='fibre_id')

# DBSCAN clustering
fiber_summary_dbscan, score_dbscan = perform_DBSCAN_clustering(fiber_sum.copy())
df_dbscan = df.merge(fiber_summary_dbscan[['fibre_id', 'cluster_id']], on='fibre_id')

# HDBSCAN clustering
fiber_summary_hdbscan, score_hdbscan = perform_HDBSCAN_clustering(fiber_sum.copy())
df_hdbscan = df.merge(fiber_summary_hdbscan[['fibre_id', 'cluster_id']], on='fibre_id')

# GMM clustering
fiber_summary_gmm, aic_gmm, bic_gmm, score_gmm = perform_gmm_clustering(fiber_sum.copy(),n)
df_gmm = df.merge(fiber_summary_gmm[['fibre_id', 'cluster_id']], on='fibre_id')

# Make a plot of the error
#fig_gmm_error = aic_bic_plot_gmm(fiber_sum.copy())

fiber_summary_agg, model, score_agg = perform_agglomerative_clustering(fiber_sum,n)
df_agg = df.merge(fiber_summary_agg[['fibre_id', 'cluster_id']], on='fibre_id')

# Make 3D plots with clusters
plot_fibers(df_k, 'K-means')
plot_fibers(df_k_pca, 'K-means with PCA')
plot_fibers(df_dbscan, 'DBSCAN')
plot_fibers(df_hdbscan, 'HDBSCAN')
plot_fibers(df_gmm, 'GMM')
plot_fibers(df_agg, 'agglomerative')

# Make score plot for all pre-defined cluster methods
score_k_list = []
score_gmm_list = []
score_agg_list = []

for n in n_clusters:
    print(f'iteration {n}')

    fiber_summary_k, inertia_k, score_k = perform_kmeans_clustering(fiber_sum,n)

    fiber_summary_gmm, aic_gmm, bic_gmm, score_gmm = perform_gmm_clustering(fiber_sum,n)

    fiber_summary_agg, model, score_agg = perform_agglomerative_clustering(fiber_sum,n)
    
    score_k_list.append(score_k)
    score_gmm_list.append(score_gmm)
    score_agg_list.append(score_agg)

plot_score(score_k_list, n_clusters, 'K-means')
plot_score(score_gmm_list, n_clusters, 'GMM')
plot_score(score_agg_list, n_clusters, 'Agglomerative')"""

ks_x_list, ks_y_list = ks_by_z_lists(df)

print("KS X:", ks_x_list)
print("KS Y:", ks_y_list)