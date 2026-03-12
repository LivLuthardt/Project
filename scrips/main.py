from data_clean import*
from ellipse import*
from tangent import*
from copula import*
from clustering import*

data_clean = data_cleaned()

df = tangent_angles(data_clean)
fiber_sum = fiber_summary(df)

#ellipse 

# copula = np.zeros((128,1))
# for row in range(1,129):
#     data_sorted = sort(df,row)
#     copula[(row-1)] = bivariate_copula(data_sorted,len(data_sorted))
# print(copula)

fiber_summary_k = perform_kmeans_clustering(fiber_sum)
# 2. Merge cluster IDs back to the original points for 3D plotting
df_clustered_k = df.merge(fiber_sum[['fibre_id', 'cluster_id']], on='fibre_id')
fig_k = plot_k(df_clustered_k)

# Use GMM clustering instead of KMeans
fiber_summary_gmm = perform_gmm_clustering(fiber_sum)
df_clustered_gmm = df.merge(fiber_sum[['fibre_id', 'cluster_id']], on='fibre_id')
fig_gmm = plot_gmm(df_clustered_gmm)

clustered, model = perform_agglomerative_clustering(df)
fig_agg = plot_agg(clustered)