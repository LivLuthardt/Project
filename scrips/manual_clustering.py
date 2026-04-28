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

#shsgs

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
def good_neighbor_distance(scaled_data):
    #Determine distance metric and store as list
    results_d = []
    for i in range(len(scaled_data)):
        for j in range(i+1, len(scaled_data)):
            #Difference in distances
            delta_x_norm = scaled_data[i, 1] - scaled_data[j, 1]
            delta_y_norm = scaled_data[i, 2] - scaled_data[j, 2]
            
            #Distance metric for distance
            D_distance = np.sqrt(delta_x_norm**2 + delta_y_norm**2)

            #Store fiber metric score with respective fibre id's
            fibre_id_i = scaled_data[i, 0]
            fibre_id_j = scaled_data[j, 0]

            results_d.append((fibre_id_i, fibre_id_j, D_distance))

    return results_d

def good_neighbor_angle(scaled_data):
    results_a = []
    for i in range(len(scaled_data)):
            for j in range(i+1, len(scaled_data)):
                #Difference in angles
                delta_anglex_norm = np.abs(scaled_data[i, 3] - scaled_data[j, 3])
                delta_angley_norm = np.abs(scaled_data[i, 4] - scaled_data[j, 4])

                #Distance metric for angle
                D_angle = np.arctan(np.sqrt(delta_anglex_norm ** 2 + delta_angley_norm ** 2))

                #Store fiber metric score with respective fibre id's
                fibre_id_i = scaled_data[i, 0]
                fibre_id_j = scaled_data[j, 0]

                results_a.append((fibre_id_i, fibre_id_j, D_angle))

    return results_a


"""Plot histogram and determine threshold"""
scores_0_d = []
layer_0_results_d = good_neighbor_distance(scaled_data)
for item in layer_0_results_d:
    scores_0_d.append(item[2])
mean_scores_0_d = np.mean(scores_0_d)
std_scores_0_d = np.std(scores_0_d)
#Threshold
n_std_d = -1
threshold_distance = mean_scores_0_d + n_std_d * std_scores_0_d
# Plot histogram
plt.hist(scores_0_d, bins=100)
plt.axvline(threshold_distance) 
plt.title("Distance Histogram") 
plt.show()
print("Mean_Distance:", mean_scores_0_d)
print("Std_Distance:", std_scores_0_d)
print("Threshold_Distance", threshold_distance)

scores_0_a = []
layer_0_results_a = good_neighbor_angle(scaled_data)
for item in layer_0_results_a:
    scores_0_a.append(item[2])
mean_scores_0_a = np.mean(scores_0_a)
std_scores_0_a = np.std(scores_0_a)
#Threshold
n_std_a = -1
threshold_angle = mean_scores_0_a + n_std_a * std_scores_0_a
# Plot histogram
plt.hist(scores_0_a, bins=100)
plt.axvline(threshold_angle)  
plt.title("Angle Histogram")
plt.show()
print("Mean_Angle:", mean_scores_0_a)
print("Std_Angle:", std_scores_0_a)
print("Threshold_Angle", threshold_angle)


"""Initialise knn and deteremine clusters + graph from this"""
results_distance = good_neighbor_distance(scaled_data)
results_angle = good_neighbor_angle(scaled_data)

#Map fibre_id to row index
fibre_ids = scaled_data[:, 0].astype(int)
id_to_idx = {fid: i for i, fid in enumerate(fibre_ids)}

#Build distance matrices
n_d = len(fibre_ids)
D_d = np.full((n_d, n_d), np.inf)
np.fill_diagonal(D_d, 0)

n_a = len(fibre_ids)
D_a = np.full((n_a, n_a), np.inf)
np.fill_diagonal(D_a, 0)

for fibre_d_i, fibre_d_j, score in results_distance:
    i = id_to_idx[int(fibre_d_i)]
    j = id_to_idx[int(fibre_d_j)]
    D_d[i, j] = score
    D_d[j, i] = score

for fibre_d_i, fibre_d_j, score in results_angle:
    ii = id_to_idx[int(fibre_d_i)]
    jj = id_to_idx[int(fibre_d_j)]
    D_a[ii, jj] = score
    D_a[jj, ii] = score

print(D_d)
print(D_a)

"""    
# apply kNN on your precomputed score matrix
knn = NearestNeighbors(n_neighbors= optimal_k, metric='precomputed')
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
plt.show()"""

