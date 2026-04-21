import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

def sklearn_neighborhood_clustering(df, n_neighbors=5, threshold=0.75):
    all_z_values = sorted(df['z'].unique())
    total_layers = len(all_z_values)
    neighbor_scores = {}

    feature_cols = ['x', 'y', 'angle_x_deg', 'angle_y_deg']
    scaler = StandardScaler()

    # --- Initial Setup (Layer 0) ---
    layer_0 = df[df['z'] == all_z_values[0]].reset_index(drop=True)
    scaled_features_0 = scaler.fit_transform(layer_0[feature_cols])
    
    nn_0 = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto')
    nn_0.fit(scaled_features_0)
    _, indices_0 = nn_0.kneighbors(scaled_features_0)

    for i, row in layer_0.iterrows():
        fiber_id = row['fibre_id']
        neighbor_scores[fiber_id] = {}
        for neighbor_idx in indices_0[i][1:]: 
            neighbor_id = layer_0.iloc[neighbor_idx]['fibre_id']
            neighbor_scores[fiber_id][neighbor_id] = 1

    # --- Check Subsequent Layers ---
    for z in all_z_values[1:]:
        current_layer = df[df['z'] == z].reset_index(drop=True)
        if current_layer.empty: continue
            
        scaled_features_current = scaler.fit_transform(current_layer[feature_cols])
        nn_current = NearestNeighbors(n_neighbors=n_neighbors + 1)
        nn_current.fit(scaled_features_current)
        _, indices_current = nn_current.kneighbors(scaled_features_current)
        
        for i, row in current_layer.iterrows():
            fiber_id = row['fibre_id']
            if fiber_id not in neighbor_scores: continue
            
            current_neighbor_ids = current_layer.iloc[indices_current[i][1:]]['fibre_id'].tolist()
            
            for orig_neighbor in neighbor_scores[fiber_id]:
                if orig_neighbor in current_neighbor_ids:
                    neighbor_scores[fiber_id][orig_neighbor] += 1

    # --- Apply Threshold ---
    results = []
    for fiber_id, neighbors in neighbor_scores.items():
        good_neighbors = [nb for nb, score in neighbors.items() if (score / total_layers) >= threshold]
        
        if len(good_neighbors) > 0:
            cluster_name = str(sorted(good_neighbors))
            results.append({'fibre_id': fiber_id, 'cluster_id': cluster_name})
        else:
            results.append({'fibre_id': fiber_id, 'cluster_id': '-1'})

    return pd.DataFrame(results)

def plot_neighborhood_clusters(df_clustered, title="Persistent Fiber Clusters (Proximity + Tilt)"):
    df_clustered['cluster_id'] = df_clustered['cluster_id'].astype(str)
    
    df_stable = df_clustered[df_clustered['cluster_id'] != '-1']
    df_unstable = df_clustered[df_clustered['cluster_id'] == '-1']

    fig = px.line_3d(df_stable,
                     x="x", y="y", z="z",
                     line_group="fibre_id",
                     color="cluster_id",
                     title=title)

    if not df_unstable.empty:
        fig.add_trace(go.Scatter3d(
            x=df_unstable['x'], y=df_unstable['y'], z=df_unstable['z'],
            mode='lines',
            line=dict(color='lightgrey', width=1),
            opacity=0.3,
            name='Unclustered',
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(aspectmode="manual", aspectratio=dict(x=15, y=7.5, z=1)),
        showlegend=False 
    )
    fig.show()