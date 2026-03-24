# OVERVIEW OF DIFFERENT METHODS: https://scikit-learn.org/stable/modules/clustering.html
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from hdbscan import HDBSCAN
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------PCA Calculation for 10 Features-----------------------------

#x_mean, y_mean, angle_x_mean, angle_y_mean, x, y, angle_x, angle_y, tilt_angle_deg     
def analyze_redundancy(corr_matrix, loadings, corr_threshold=0.9, loading_diff_threshold=0.15):
    print("\n--- Highly Correlated Feature Pairs ---")
    correlated_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            f1 = corr_matrix.columns[i]
            f2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]

            if abs(corr_val) >= corr_threshold:
                correlated_pairs.append((f1, f2, corr_val))
                print(f"{f1} ↔ {f2}: correlation = {corr_val:.2f}")

    print("\n--- Similar PCA Loading Pairs ---")
    loading_pairs = []

    for i in range(len(loadings.index)):
        for j in range(i + 1, len(loadings.index)):
            f1 = loadings.index[i]
            f2 = loadings.index[j]

            vec1 = loadings.loc[f1].values
            vec2 = loadings.loc[f2].values

            if np.all(np.abs(np.abs(vec1) - np.abs(vec2)) < loading_diff_threshold):
                loading_pairs.append((f1, f2))
                print(f"{f1} ↔ {f2}: similar PCA loading pattern")

    print("\n--- Strong Redundancy Candidates ---")
    for f1, f2, corr_val in correlated_pairs:
        for g1, g2 in loading_pairs:
            if {f1, f2} == {g1, g2}:
                print(f"{f1} and {f2} are strong redundancy candidates")

def PCA_determination(df):
    features = ['x_mean', 'y_mean', 'angle_x_mean', 'angle_y_mean', 'x', 'y', 'angle_x_deg', 'angle_y_deg', 'tilt_angle_deg', 'tilt_angle_mean']

    scale = StandardScaler()
    data_scaled = scale.fit_transform(df[features])

    corr_matrix = pd.DataFrame(data_scaled, columns=features).corr()
    print("\nCorrelation matrix:\n", corr_matrix.to_string())

    pca = PCA(n_components=data_scaled.shape[1])
    data_transformed = pca.fit_transform(data_scaled)

    loadings = pd.DataFrame(pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=features)

    print("\nPCA Loadings:\n", loadings.to_string())

    analyze_redundancy(corr_matrix, loadings)

    coverage_lst = np.cumsum(pca.explained_variance_ratio_) * 100

    plt.figure()
    plt.plot(
        np.arange(1, data_transformed.shape[1] + 1),
        coverage_lst,
        marker='o',
        label='Cumulative explained variance'
    )
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Coverage (%)')
    plt.title('PCA Coverage vs Number of Principal Components')
    plt.xlim(1, data_transformed.shape[1])
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    #plt.savefig("Correct_PCA_10_coverage.png")
    plt.close()
    print("PCA plot saved")

    return pca, data_transformed, coverage_lst

# ---------------------------------------METHOD 1: k means------------------------------------------

def perform_kmeans_clustering(df, n_clusters):
    df = df.drop_duplicates(subset=['fibre_id'])
    features = ['x_mean', 'y_mean', 'tilt_angle_mean']

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    

    scaled_df = pd.DataFrame(scaled_data, columns=features) #trying to give them more importance
    #scaled_df['x_mean'] *= 1
    #scaled_df['y_mean'] *= 1

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_df.values)

    df['cluster_id'] = cluster_labels

    score = calinski_harabasz_score(scaled_df, cluster_labels)
    return df, kmeans.inertia_, score

# ------------------------------------METHOD 1B: K-MEANS WITH PCA---------------------------

def perform_kmeans_clustering_with_pca(df, n_clusters, n_components=3):
    df = df.drop_duplicates(subset=['fibre_id'])
    features = ['x_mean', 'y_mean', 'tilt_angle_mean']
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)

    kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans_pca.fit_predict(pca_data)

    df_pca_clustered = df.copy()
    df_pca_clustered['cluster_id'] = cluster_labels

    score_pca = calinski_harabasz_score(pca_data, cluster_labels)

    return df_pca_clustered, kmeans_pca.inertia_, score_pca, pca.explained_variance_ratio_

# -----------------------------------------METHOD 2: DBSCAN ----------------------------------------------
def perform_DBSCAN_clustering(df):
    df = df.drop_duplicates(subset=['fibre_id'])
    # Features to use for clustering
    features = ['x_mean', 'y_mean', 'tilt_angle_mean']

    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # Fit HDSCAN
    DBS = DBSCAN()
    cluster_labels = DBS.fit_predict(scaled_data)

    # Add cluster labels to DataFrame
    df['cluster_id'] = cluster_labels

    # Calculate scores
    score = calinski_harabasz_score(scaled_data, cluster_labels)
    print(f'Calinski-Harabasz score of DBSCAN is {score}')
    return df,score
    
#-----------------------------------------METHOD 3: HDBSCAN-----------------------------------------------
def perform_HDBSCAN_clustering(df):
    df = df.drop_duplicates(subset=['fibre_id'])
    # Features to use for clustering
    features = ['x_mean', 'y_mean', 'tilt_angle_mean']

    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    # Fit HDSCAN
    HDBS = HDBSCAN()
    cluster_labels = HDBS.fit_predict(scaled_data)

    # Add cluster labels to DataFrame
    df['cluster_id'] = cluster_labels

    # Calculate calinski-harabasz score
    score = calinski_harabasz_score(scaled_data, cluster_labels)
    print(f'Calinski-Harabasz score of DBSCAN is {score}')
    return df,score

# ---------------------------------------METHOD 4: Gaussian Mixture GMM-----------------------------------
def perform_gmm_clustering(df,n_clusters):
    df = df.drop_duplicates(subset=['fibre_id'])
    # Features to use for clustering
    features = ['x_mean', 'y_mean', 'tilt_angle_mean']
    
    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    cluster_labels = gmm.fit_predict(scaled_data)
    
    # Add cluster labels to DataFrame
    df['cluster_id'] = cluster_labels

    # Calculate calinski-harabasz score
    score = calinski_harabasz_score(scaled_data, cluster_labels)
    return df,gmm.aic(scaled_data),gmm.bic(scaled_data),score

# ------------------------------------METHOD 5: Agglomerative (Hierarchical)------------------------------
def perform_agglomerative_clustering(df, n_clusters):
    df = df.drop_duplicates(subset=['fibre_id'])
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

    # Calculate scores
    score = calinski_harabasz_score(X_scaled, labels)
    return df_clustered, model, score

# ------------------------------------Plots----------------------------------------------------------------------

def sse_plot_kmeans_pca(df, n_components=3):
    sse_pca = []
    n_clusters_range = range(1, 11)

    for k in n_clusters_range:
        _, inertia_pca, _, _ = perform_kmeans_clustering_with_pca(
            df,
            n_clusters=k,
            n_components=n_components
        )
        sse_pca.append(inertia_pca)

    plot_df_pca = pd.DataFrame({
        'Number of Clusters': n_clusters_range,
        'SSE': sse_pca
    })

    fig = px.line(
        plot_df_pca,
        x='Number of Clusters',
        y='SSE',
        markers=True,
        title=f"SSE vs Number of Clusters (K-means with PCA, {n_components} PCs)"
    )

    fig.show()

def plot_fibers(clustered,title):
    fig_3d = px.line_3d(
        clustered, 
        x='x', y='y', z='z', 
        color='cluster_id',
        line_group='fibre_id',
        title=title
    )
    #fig_3d.show()
    print(f'Plot {title} finished')

def plot_score(df, n_clusters):
    score_list_k = []
    score_list_gmm = []
    score_list_agg = []
    for n in n_clusters:
        _,_, score_k = perform_kmeans_clustering(df,n)
        _,_,_, score_gmm = perform_gmm_clustering(df,n)
        _,_, score_agg = perform_agglomerative_clustering(df,n)
    
        score_list_k.append(score_k)
        score_list_gmm.append(score_gmm)
        score_list_agg.append(score_agg)


    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Number of Clusters': list(n_clusters) * 3,
        'Calinski-Harabasz': score_list_k + score_list_gmm + score_list_agg,
        'Method': ['K-means'] * len(n_clusters) + ['GMM'] * len(n_clusters) + ['Agglomerative'] * len(n_clusters)
    })

    # Create 2D line plot with Plotly Express, coloring by Method
    fig = px.line(
        plot_df,
        x='Number of Clusters',
        y='Calinski-Harabasz',
        color='Method',
        markers=True,
        title="Calinski-Harabasz score vs Number of Clusters for Clustering Methods"
    )

    #fig.show()
    fig.write_image(f"CD_score_plot.png")
    print(f'Plot CD finished')

def plot_sse_k(df, n_clusters):
    sse = []
    for k in n_clusters:
        _ , inertia, _ = perform_kmeans_clustering(df,n_clusters=k)
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
        title="SSE vs Number of Clusters for K-means"
    )

    #fig.show()
    fig.write_image(f"SSE_k_means.png")
    print(f'Plot SSE K-means finished')

def plot_aic_bic_gmm(df, n_clusters):
    aic_vals = []
    bic_vals = []
    for k in n_clusters:
        _ , aic , bic, _ = perform_gmm_clustering(df,n_clusters=k)
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
        title="AIC and BIC vs Number of Clusters for GMM",
        labels={'Value': 'Criterion Value'}
    )

    #fig.show()
    fig.write_image(f"AIC_BIC_GMM.png")
    print(f'Plot AIC BIC GMM finished')