from data_clean import*
from ellipse import*
from tangent import*
from copula import*
from clustering import*

df = pd.read_csv('raw_data.csv')
data_clean = data_cleaned(df)
df = tangent_angles(data_clean)
fiber_sum,n_fibers = fiber_summary(df)

#ellipse 
xtiltAngles, ytiltAngles = [], [] #Init empty lists
for r in df.itertuples(index=True):
    x2 = (r[3], r[4], r[2]) #Current fiber point
    if r[0] == 0: tilt = (0, 0) #Can't compute tilt from a single point
    else:  tilt = eTiltAngles(x1, x2) #Pass the past and current points
    xtiltAngles.append(tilt[0])
    ytiltAngles.append(tilt[1])
    x1 = x2 #Set the current point to the past point
df = df.assign(EllipseXTilt = xtiltAngles, EllipseYTilt = ytiltAngles) #Add the tilt angles as a df column
df = df.dropna(subset=['dx', 'dy', 'dz'])
print(df)

copula_lst = [0 for _ in range(129)]
# print(df)


# 129 is the amount of z values
# n_fibers is the amount of unique fibers
# 2 is the amount of parameters we can put in our copula
data_sim_lst = np.empty((129,n_fibers,2))

# list to contain copulas 
# TODO remove the 0 inside here once we have 129 istead of 128 datapoints
cop_lst = [0]

# Iterate [1,128] because for z = 0 certain parameters like dx and dy are undefined
# TODO once the dataframe is changed to account for z = 0 we can do range(129)
for row_n in range(1,129):
    # print(f'Iterating over z = {row_n}')
    data_filtered = sort(df,row_n,'angle_x_deg','angle_y_deg')
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

# Plot covariance of Gaussian copulas
# plot_cop_parameters(cop_lst)

fiber_summary_k = perform_kmeans_clustering(fiber_sum,5)
# 2. Merge cluster IDs back to the original points for 3D plotting
df_clustered_k = df.merge(fiber_sum[['fibre_id', 'cluster_id']], on='fibre_id')
fig_k = plot_k(df_clustered_k)
# make a plot of the error
fig_k_error = sse_plot_k(fiber_sum)


n = 5

# Use GMM clustering instead of KMeans
fiber_summary_gmm = perform_gmm_clustering(fiber_sum,n)
df_clustered_gmm = df.merge(fiber_sum[['fibre_id', 'cluster_id']], on='fibre_id')
fig_gmm = plot_gmm(df_clustered_gmm)
# make a plot of the error
fig_gmm_error = aic_bic_plot_gmm(fiber_sum)

clustered, model = perform_agglomerative_clustering(df)
fig_agg = plot_agg(clustered)

df['tilt_angle_deg'] = np.degrees(df['tilt_angle_rad'])
max_tilts = df.groupby('fibre_id')['tilt_angle_deg'].max()
# Dynamically set threshold to the 99th percentile (keeps 99%, drops top 1%)
threshold = max_tilts.quantile(0.99)
fig_pdf = px.histogram(max_tilts, nbins=100, histnorm='probability density')
fig_pdf.show()