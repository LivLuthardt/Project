from data_clean import*
from ellipse import*
from tangent import*
from copula import*
from clustering import*

df = pd.read_csv('raw_data.csv')
data_clean = data_cleaned(df)
df = tangent_angles(data_clean)
fiber_sum = fiber_summary(df)

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

copula_lst = [0 for _ in range(129)]
# Iterate [1,128] because for z = 0 certain parameters like dx and dy are undefined
for row_n in range(1,129):
    data_filtered = sort(df,row_n,'angle_x_deg','angle_y_deg')
    copula_lst[row_n-1],cop = bivariate_copula(data_filtered,len(data_filtered))

    if row_n == 67:
        cop.show()

print(copula_lst)

fiber_summary_k = perform_kmeans_clustering(fiber_sum,5)
# 2. Merge cluster IDs back to the original points for 3D plotting
df_clustered_k = df.merge(fiber_sum[['fibre_id', 'cluster_id']], on='fibre_id')
fig_k = plot_k(df_clustered_k)
# make a plot of the sse to evaluate the accuracy of the method
sse_k = sse_plot_k(fiber_sum)
sse_k.show()

# Use GMM clustering instead of KMeans
fiber_summary_gmm = perform_gmm_clustering(fiber_sum)
df_clustered_gmm = df.merge(fiber_sum[['fibre_id', 'cluster_id']], on='fibre_id')
fig_gmm = plot_gmm(df_clustered_gmm)

clustered, model = perform_agglomerative_clustering(df)
fig_agg = plot_agg(clustered)