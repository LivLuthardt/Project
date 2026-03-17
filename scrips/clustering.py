# OVERVIEW OF DIFFERENT METHODS: https://scikit-learn.org/stable/modules/clustering.html
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from hdbscan import HDBSCAN
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage

# ---------------------------------------METHOD 1: k means------------------------------------------
     
def perform_kmeans_clustering(df, n_clusters):
    features = ['x_mean', 'y_mean', 'angle_x_mean', 'angle_y_mean']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features) #trying to give them more importance
    scaled_df['x_mean'] *= 1.5
    scaled_df['y_mean'] *= 1.5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_df.values)
    df['cluster_id'] = cluster_labels
    score = silhouette_score(scaled_df.values, cluster_labels)
    return df, kmeans.inertia_, score

def sse_plot_k(df):
    sse = []
    n_clusters = range(1,11)
    for k in n_clusters:
        df , inertia = perform_kmeans_clustering(df,n_clusters=k)
        sse.append(inertia)

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Number of Clusters': n_clusters,
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
# -----------------------------------------METHOD 2: DBSCAN ----------------------------------------------
def perform_DBSCAN_clustering(df):
    # Features to use for clustering
    features = ['x_mean', 'y_mean', 'angle_x_mean', 'angle_y_mean']

    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    # Fit HDSCAN
    DBS = DBSCAN()
    cluster_labels = DBS.fit_predict(scaled_data)

    # Add cluster labels to DataFrame
    df['cluster_id'] = cluster_labels

    # Calculate silhouette score
    score = silhouette_score(scaled_data, cluster_labels)
    return df
    
#-----------------------------------------METHOD 3: HDBSCAN-----------------------------------------------
def perform_HDBSCAN_clustering(df):
    # Features to use for clustering
    features = ['x_mean', 'y_mean', 'angle_x_mean', 'angle_y_mean']

    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    # Fit HDSCAN
    HDBS = HDBSCAN()
    cluster_labels = HDBS.fit_predict(scaled_data)

    # Add cluster labels to DataFrame
    df['cluster_id'] = cluster_labels

    # Calculate silhouette score
    score = silhouette_score(scaled_data, cluster_labels)
    return df

# ---------------------------------------METHOD 4: Gaussian Mixture GMM-----------------------------------
def perform_gmm_clustering(df,n_clusters):
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

    # Calculate silhouette score
    score = silhouette_score(scaled_data, cluster_labels)
    return df,gmm.aic(scaled_data),gmm.bic(scaled_data),score

def aic_bic_plot_gmm(df):
    aic_vals = []
    bic_vals = []
    n_clusters = range(1,11)
    for k in n_clusters:
        df , aic , bic = perform_gmm_clustering(df,n_clusters=k)
        aic_vals.append(aic)
        bic_vals.append(bic)

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Number of Clusters': list(n_clusters) * 2,
        'Criterion': ['AIC'] * len(n_clusters) + ['BIC'] * len(n_clusters),
        'Value': aic_vals + bic_vals
    })

    fig = px.line(
        plot_df, 
        x='Number of Clusters', 
        y='Value', 
        color='Criterion',
        markers=True,
        title="AIC and BIC vs Number of Clusters",
        labels={'Value': 'Criterion Value'}
    )

    fig.show()

# ------------------------------------METHOD 5: Agglomerative (Hierarchical)------------------------------
def perform_agglomerative_clustering(df, n_clusters):
    df_sorted = df.sort_values(['fibre_id', 'z']).copy()
    features = ['x', 'y', 'tilt_angle_deg']

    fiber_vectors = (
        df_sorted
        .groupby('fibre_id')[features]
        .apply(lambda g: g.to_numpy().flatten())
    )

    X = np.vstack(fiber_vectors.values)
    fiber_ids = fiber_vectors.index

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )

    labels = model.fit_predict(X_scaled)

    fibre_to_cluster = dict(zip(fiber_ids, labels))

    df_clustered = df.copy()
    df_clustered['cluster_id'] = df_clustered['fibre_id'].map(fibre_to_cluster)

    # Calculate silhouette score
    score = silhouette_score(X_scaled, labels)
    return df_clustered, model, score

# ------------------------------------Plots----------------------------------------------------------------------

def plot_fibers(clustered,title):
    fig_3d = px.line_3d(
        clustered, 
        x='x', y='y', z='z', 
        color='cluster_id',
        line_group='fibre_id',
        title=title
    )
    fig_3d.show()

def plot_silhouette(score, n_clusters):
    sse = []
    n_clusters = range(1,11)
    for k in n_clusters:
        df , inertia = perform_kmeans_clustering(df,n_clusters=k)
        sse.append(inertia)

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Number of Clusters': n_clusters,
        'Silhouette': score
    })

    # Create 2D line plot with Plotly Express
    fig = px.line(
        plot_df, 
        x='Number of Clusters', 
        y='SSE', 
        markers=True,
        title="Silhouette vs Number of Clusters"
    )

    fig.show()