import pandas as pd 
import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from data_clean import data_cleaned
from tangent import tangent_angles_central
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN



"""Ïmport data and manipulate dataframe"""
raw_df = pd.read_csv('raw_data.csv')
data_clean = data_cleaned(raw_df)
df = tangent_angles_central(data_clean)

layer_0 = df[df['z_idx'] == 0]
layer_0 = layer_0.reset_index(drop=True)

features = ['fibre_id', 'x', 'y', 'angle_x_deg', 'angle_y_deg']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(layer_0[['x', 'y', 'angle_x_deg', 'angle_y_deg']])
scaled_data = np.column_stack((layer_0['fibre_id'].values, scaled_data))


""" --- Optimal n_neighbors (Elbow Method) --- """
X_eval = scaled_data[:, 1:] # Exclude fibre_id

avg_distances = []
k_range = range(1, 50) 

for k in k_range:
    knn_eval = NearestNeighbors(n_neighbors=k)
    knn_eval.fit(X_eval)
    distances, _ = knn_eval.kneighbors(X_eval)
    avg_distances.append(np.mean(distances[:, -1]))

plt.figure(figsize=(10,6))
plt.plot(k_range, avg_distances, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=6)
plt.title('Average K-Distance vs. K Value (Find the Elbow)')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Average Distance to K-th Neighbor')
plt.show()

from kneed import KneeLocator

kneedle = KneeLocator(k_range, avg_distances, S=1.0, curve='concave', direction='increasing')
optimal_k = kneedle.knee
print(f"The optimal number of neighbors is: {optimal_k}")
""" ------------------------------------------ """

"""Neighborhood rule"""
def good_neighbor(scaled_data):
    #Determine distance metric and store as list
    results = []
    for i in range(len(scaled_data)):
        for j in range(i+1, len(scaled_data)):
            #Difference in angles and distances
            delta_x_norm = scaled_data[i, 1] - scaled_data[j, 1]
            delta_y_norm = scaled_data[i, 2] - scaled_data[j, 2]
            delta_anglex_norm = scaled_data[i, 3] - scaled_data[j, 3]
            delta_angley_norm = scaled_data[i, 4] - scaled_data[j, 4]

            #Distance metric for knn
            weight_d = 1
            weight_a = 1
            D_distance = np.sqrt(delta_x_norm**2 + delta_y_norm**2)
            D_angle = np.sqrt(delta_anglex_norm**2 + delta_angley_norm**2)
            D_knn = weight_a * D_angle + weight_d * D_distance

            #Store fiber metric score with respective fibre id's
            fibre_id_i = scaled_data[i, 0]
            fibre_id_j = scaled_data[j, 0]

            results.append((fibre_id_i, fibre_id_j, D_knn))

    return results

"""Plot histogram and determine threshold"""
scores_0 = []
layer_0_results = good_neighbor(scaled_data)
for item in layer_0_results:
    scores_0.append(item[2])
mean_scores_0 = np.mean(scores_0)
std_scores_0 = np.std(scores_0)
#Threshold
n_std = 0
threshold = mean_scores_0 + n_std * std_scores_0
# Plot histogram
plt.hist(scores_0, bins=100)
plt.axvline(threshold)  
plt.show()
print("Mean:", mean_scores_0)
print("Std:", std_scores_0)
print("Threshold", threshold)

   
"""Initialise knn on the initial layer (layer 0) and plot the graph knn_graph"""
"""
X = scaled_data[:, 1:]
A = kneighbors_graph(X, n_neighbors=5, mode='distance')
A = A.minimum(A.T)
A.data[A.data > threshold] = 0
A.eliminate_zeros()
G = nx.from_scipy_sparse_array(A)
pos = {i: (scaled_data[i, 1], scaled_data[i, 2]) for i in range(len(scaled_data))}
nx.draw(G, pos, node_size=5, width=0.1, alpha=0.3)
plt.show()
print(nx.number_connected_components(G))
"""

"""Initialise knn and deteremine clusters + graph from this"""
results = good_neighbor(scaled_data)

#Map fibre_id to row index
fibre_ids = scaled_data[:, 0].astype(int)
id_to_idx = {fid: i for i, fid in enumerate(fibre_ids)}

#Build distance matrix
n = len(fibre_ids)
D = np.full((n, n), np.inf)
np.fill_diagonal(D, 0)

for fibre_d_i, fibre_d_j, score in results:
    i = id_to_idx[int(fibre_d_i)]
    j = id_to_idx[int(fibre_d_j)]
    D[i, j] = score
    D[j, i] = score

# apply kNN on your precomputed score matrix
knn = NearestNeighbors(n_neighbors=100, metric='precomputed')
knn.fit(D)

distances, indices = knn.kneighbors(D)

# store accepted kNN neighbor relations
knn_results = []

for i in range(len(indices)):
    fid_i = int(fibre_ids[i])

    for j in range(1, len(indices[i])):   # skip self
        neighbor_idx = indices[i, j]
        fid_j = int(fibre_ids[neighbor_idx])
        score = distances[i, j]

        if score <= threshold:
            knn_results.append((fid_i, fid_j, score))

# keep only mutual kNN edges
edge_dict = {(fid_i, fid_j): score for fid_i, fid_j, score in knn_results}

mutual_results = []
for fid_i, fid_j, score in knn_results:
    if (fid_j, fid_i) in edge_dict:
        mutual_results.append((fid_i, fid_j, score))

# build graph
G = nx.Graph()

for fid in fibre_ids:
    G.add_node(int(fid))

for fid_i, fid_j, score in mutual_results:
    G.add_edge(int(fid_i), int(fid_j), weight=score)

# connected components = clusters
clusters = [sorted(list(c)) for c in nx.connected_components(G)]

print("Amount of clusters:", len(clusters))
print("Cluster sizes:", [len(c) for c in clusters])

for i, cluster in enumerate(clusters, start=1):
    print(f"Cluster {i}: {cluster}")

# plot graph
pos = {int(scaled_data[i, 0]): (scaled_data[i, 1], scaled_data[i, 2]) for i in range(len(scaled_data))}

nx.draw(G, pos, node_size=8, width=0.2, alpha=0.5, with_labels=False)
plt.show()