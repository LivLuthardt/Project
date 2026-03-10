# OVERVIEW OF DIFFERENT METHODS: https://scikit-learn.org/stable/modules/clustering.html
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tangent import df_final  
from tangent import fiber_summary
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# ---------------------------------------METHOD 1: k means------------------------------------------
     
def perform_kmeans_clustering(summary_df, n_clusters=5):
    features = ['x_mean', 'y_mean', 'angle_x_mean', 'angle_y_mean']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(summary_df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features) #trying to give them more importance
    scaled_df['x_mean'] *= 1.5
    scaled_df['y_mean'] *= 1.5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    summary_df['cluster_id'] = kmeans.fit_predict(scaled_data)
    return summary_df

fiber_summary = perform_kmeans_clustering(fiber_summary)

# 2. Merge cluster IDs back to the original points for 3D plotting
df_clustered = df_final.merge(fiber_summary[['fibre_id', 'cluster_id']], on='fibre_id')

# 3. Plotting (Now df_clustered has x, y, z AND cluster_id)
fig = px.line_3d(
    df_clustered, 
    x='x', y='y', z='z', 
    color='cluster_id',
    line_group='fibre_id',
    title="Spatial Coherence of Fiber Clusters"
)
fig.show()

# -----------------------------------------METHOD 2: DBSCAN ----------------------------------------------


#-----------------------------------------METHOD 3: HDBSCAN-----------------------------------------------



# ---------------------------------------METHOD 4: Gaussian Mixture GMM-----------------------------------

from sklearn.mixture import GaussianMixture

def perform_gmm_clustering(df, n_components=2):
    # Features to use for clustering
    features = ['angle_x_deg','angle_y_deg']
    
    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    cluster_labels = gmm.fit_predict(scaled_data)
    
    # Add cluster labels to DataFrame
    df['cluster_id'] = cluster_labels
    
    return df

# Use GMM clustering instead of KMeans
df_clustered = perform_gmm_clustering(df_final)

# Plot clusters
fig = px.scatter(
    df_clustered, 
    x='angle_x_deg', 
    y='angle_y_deg', 
    color='cluster_id',
    title="Fiber Clusters by Planar Tilt",
    labels={'angle_x_deg': 'ZX Planar Tilt [deg]', 'angle_y_deg': 'ZY Planar Tilt [deg]'}
)
fig.show()

fig_3d = px.line_3d(
    df_clustered, 
    x='x', y='y', z='z', 
    color='cluster_id',
    line_group='fibre_id',
    title="Spatial Coherence of Fiber Clusters"
)
fig_3d.show()


# ------------------------------------METHOD 5: Agglomerative (Hierarchical)------------------------------

def Aggl_clustering(X):
    
    features = ['x', 'y', 'tilt_angle_deg']
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(X[features])
    
    model = AgglomerativeClustering(linkage = 'ward', distance_threshold = 0, n_clusters = None)
    model.fit(data_scaled)

    







