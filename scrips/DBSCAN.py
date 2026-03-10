# OVERVIEW OF DIFFERENT METHODS: https://scikit-learn.org/stable/modules/clustering.html


# ---------------------------------------METHOD 1: k means------------------------------------------
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tangent import df_final  
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


def perform_kmeans_clustering(df, n_clusters=6):
    #'angle_x_deg', 'angle_y_deg',
    features = ['x', 'y', 'tilt_angle_deg']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(scaled_data)
    
    return df

df_clustered = perform_kmeans_clustering(df_final)


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

# -----------------------------------------METHOD 2: DBSCAN ----------------------------------------------


#-----------------------------------------METHOD 3: HDBSCAN-----------------------------------------------



# ---------------------------------------METHOD 4: Gaussian Mixture GMM-----------------------------------



# ------------------------------------METHOD 5: Agglomerative (Hierarchical)------------------------------

def Aggl_clustering(df)
    linkage_model = AgglomerativeClustering(linkage = 'single', distance_threshold = 0, n_clusters = None)

    







