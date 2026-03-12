# OVERVIEW OF DIFFERENT METHODS: https://scikit-learn.org/stable/modules/clustering.html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------METHOD 1: k means------------------------------------------
     
def perform_kmeans_clustering(df, n_clusters):
    features = ['x_mean', 'y_mean', 'angle_x_mean', 'angle_y_mean']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features) #trying to give them more importance
    scaled_df['x_mean'] *= 1.5
    scaled_df['y_mean'] *= 1.5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(scaled_data)
    return df, kmeans.inertia_

def sse_plot_k(df):
    sse = []
    for k in range(1, 11):
        df , inertia = perform_kmeans_clustering(df,n_clusters=k)
        sse.append(inertia)

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Number of Clusters': range(1, 11),
        'SSE': sse
    })

    # Create 2D line plot with Plotly Express
    fig = px.line(
        plot_df, 
        x='Number of Clusters', 
        y='SSE', 
        markers=True,
        title="SSE vs Number of Clusters"
    )

    fig.show()

def plot_k(df_clustered):
    # 3. Plotting (Now df_clustered has x, y, z AND cluster_id)
    fig = px.line_3d(
        df_clustered, 
        x='x', y='y', z='z', 
        color='cluster_id',
        line_group='fibre_id',
        title="K-means "
    )
    fig.show()

# -----------------------------------------METHOD 2: DBSCAN ----------------------------------------------


#-----------------------------------------METHOD 3: HDBSCAN-----------------------------------------------



# ---------------------------------------METHOD 4: Gaussian Mixture GMM-----------------------------------

from sklearn.mixture import GaussianMixture

def perform_gmm_clustering(df,n_clusters=5):
    # Features to use for clustering
    features = ['x_mean', 'y_mean', 'angle_x_mean', 'angle_y_mean']
    
    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    cluster_labels = gmm.fit_predict(scaled_data)
    
    # Add cluster labels to DataFrame
    df['cluster_id'] = cluster_labels
    
    return df


# Plot clusters
def plot_gmm(df_clustered):
    fig = px.line_3d(
        df_clustered, 
        x='x', y='y', z='z', 
        color='cluster_id',
        line_group='fibre_id',
        title="GMM"
    )
    fig.show()


# ------------------------------------METHOD 5: Agglomerative (Hierarchical)------------------------------

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

def perform_agglomerative_clustering(X, n_clusters=5):
    features = ['x', 'y', 'tilt_angle_deg']
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X[features])
    
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    
    X = X.copy()
    X['cluster_id'] = model.fit_predict(scaled_data)
    
    return X, model

def plot_agg(clustered):
    fig_3d = px.line_3d(
        clustered, 
        x='x', y='y', z='z', 
        color='cluster_id',
        line_group='fibre_id',
        title="AGGL"
    )
    fig_3d.show()







