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
from kneed import KneeLocator
import matplotlib.patches as mpatches

"""Import data and manipulate dataframe"""
raw_df = pd.read_csv('raw_data.csv')
data_clean = data_cleaned(raw_df)
df = tangent_angles_central(data_clean)

layer_0 = df[df['z_idx'] == 0]
layer_0 = layer_0.reset_index(drop=True)

features = ['fibre_id', 'x', 'y', 'angle_x_deg', 'angle_y_deg']

cleaned_data = layer_0[['x', 'y', 'angle_x_deg', 'angle_y_deg']]
cleaned_data = np.column_stack((layer_0['fibre_id'].values, cleaned_data))

#For all other layers create dataframe
cleaned_data_i = []
unique_layers = sorted(df['z_idx'].unique())
for layer_i in unique_layers[1:]:
    current_layer = df[df['z_idx'] == layer_i]
    current_layer = current_layer.reset_index(drop=True)
    layer_features = current_layer[['fibre_id','x', 'y', 'angle_x_deg', 'angle_y_deg']]
    cleaned_data_i.append(layer_features)


""" --- Optimal n_neighbors (Elbow Method) --- """
X_eval = cleaned_data[:, 1:] # Exclude fibre_id

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
#plt.show()
plt.savefig(fname = 'Elbow')
plt.close('all')

kneedle = KneeLocator(k_range, avg_distances, S=1.0, curve='concave', direction='increasing')
optimal_k = kneedle.knee
print(f"The optimal number of neighbors is: {optimal_k}")
""" ------------------------------------------ """


"""Neighborhood rule"""
def good_neighbor_distance(cleaned_data):
    #Determine distance metric and store as list
    results_d = []
    for i in range(len(cleaned_data)):
        for j in range(i+1, len(cleaned_data)):
            #Difference in distances
            delta_x_norm = cleaned_data[i, 1] - cleaned_data[j, 1]
            delta_y_norm = cleaned_data[i, 2] - cleaned_data[j, 2]
            
            #Distance metric for distance
            D_distance = np.sqrt(delta_x_norm**2 + delta_y_norm**2)

            #Store fiber metric score with respective fibre id's
            fibre_id_i = cleaned_data[i, 0]
            fibre_id_j = cleaned_data[j, 0]

            results_d.append((fibre_id_i, fibre_id_j, D_distance))

    return results_d

def good_neighbor_angle(cleaned_data):
    results_a = []
    for i in range(len(cleaned_data)):
            for j in range(i+1, len(cleaned_data)):
                #Difference in angles
                delta_anglex_norm = np.abs(cleaned_data[i, 3] - cleaned_data[j, 3])
                delta_angley_norm = np.abs(cleaned_data[i, 4] - cleaned_data[j, 4])

                #Distance metric for angle
                D_angle = np.arctan(np.sqrt(delta_anglex_norm ** 2 + delta_angley_norm ** 2))

                #Store fiber metric score with respective fibre id's
                fibre_id_i = cleaned_data[i, 0]
                fibre_id_j = cleaned_data[j, 0]

                results_a.append((fibre_id_i, fibre_id_j, D_angle))

    return results_a


"""Plot histogram and determine threshold"""
scores_0_d = []
layer_0_results_d = good_neighbor_distance(cleaned_data)
for item in layer_0_results_d:
    scores_0_d.append(item[2])
mean_scores_0_d = np.mean(scores_0_d)
std_scores_0_d = np.std(scores_0_d)
#Threshold
n_std_d = 1
threshold_distance = mean_scores_0_d + n_std_d * std_scores_0_d
# Plot histogram
plt.figure()
plt.hist(scores_0_d, bins=100)
plt.axvline(threshold_distance,color = 'r', label = 'Threshold') 
plt.title("Distance Histogram")
plt.xlabel("Distance between pairs of points")
plt.ylabel("Frequency")
plt.legend()
plt.savefig(fname = 'Distance_Histogram')
plt.close('all')
print("Mean_Distance:", mean_scores_0_d)
print("Std_Distance:", std_scores_0_d)
print("Threshold_Distance", threshold_distance)

scores_0_a = []
layer_0_results_a = good_neighbor_angle(cleaned_data)
for item in layer_0_results_a:
    scores_0_a.append(item[2])
mean_scores_0_a = np.mean(scores_0_a)
std_scores_0_a = np.std(scores_0_a)
#Threshold
n_std_a = 2
threshold_angle = mean_scores_0_a + n_std_a * std_scores_0_a
# Plot histogram
plt.figure()
plt.hist(scores_0_a, bins=100)
plt.axvline(threshold_angle,color = 'r', label = 'Threshold') 
plt.title("Angle Histogram")
plt.xlabel("Angle bwetween pairs of points")
plt.ylabel("Frequency")
plt.legend()
plt.savefig(fname = 'Angle_Histogram')
plt.close('all')
print("Mean_Angle:", mean_scores_0_a)
print("Std_Angle:", std_scores_0_a)
print("Threshold_Angle", threshold_angle)


"""Initialise knn and deteremine clusters + graph from this"""
results_distance = good_neighbor_distance(cleaned_data)
results_angle = good_neighbor_angle(cleaned_data)

#Map fibre_id to row index
fibre_ids = cleaned_data[:, 0].astype(int)
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


""""Build combined graph"""
G_both = nx.Graph()
G_both.add_nodes_from([int(fid) for fid in fibre_ids])

knn_d = NearestNeighbors(n_neighbors=optimal_k, metric='precomputed')
knn_d.fit(D_d)

distances_d, indices_d = knn_d.kneighbors(D_d)

for i in range(len(indices_d)):
    fid_i = int(fibre_ids[i])

    for jj in range(1, len(indices_d[i])):  #Skip self
        neighbor_idx = indices_d[i, jj]
        fid_j = int(fibre_ids[neighbor_idx])

        score_d = D_d[i, neighbor_idx]
        score_a = D_a[i, neighbor_idx]

        if score_d <= threshold_distance and score_a <= threshold_angle:
            combined_score = score_d + score_a
            similarity = 1 / (combined_score + 1e-12)
            G_both.add_edge(fid_i, fid_j, weight=similarity)

print("Combined graph nodes:", G_both.number_of_nodes())
print("Combined graph edges:", G_both.number_of_edges())
print("Isolated nodes:", len(list(nx.isolates(G_both))))

G_cluster = G_both.copy()

#Remove isolated fibres
G_cluster.remove_nodes_from(list(nx.isolates(G_cluster)))

#Creates clusters based on densely populated nodes
communities = nx.community.greedy_modularity_communities(G_cluster, weight="weight")
clusters = [sorted(list(c)) for c in communities]

#Remove tiny clusters
min_cluster_size = 2
clusters = [c for c in clusters if len(c) >= min_cluster_size]

print("Amount of clusters:", len(clusters))
print("Cluster sizes:", [len(c) for c in clusters])

for i, cluster in enumerate(clusters, start=1):
    print(f"Cluster {i}: {cluster}")


#Create a dictionary to map each node to its cluster
node_to_cluster = {}
for cluster_id, cluster in enumerate(clusters):
    for node in cluster:
        node_to_cluster[node] = cluster_id

#Assign isolated nodes to a default cluster (e.g., -1)
isolated_nodes = list(nx.isolates(G_both))
for node in isolated_nodes:
    node_to_cluster[node] = -1  # Default cluster for isolated nodes

#Assign a color to each cluster (including the default cluster)
num_clusters = len(clusters)
colors = plt.cm.tab20(np.linspace(0, 1, num_clusters + 1))  # +1 for the default cluster

#Create a list of node colors based on their cluster
node_colors = [colors[node_to_cluster[node]] for node in G_both.nodes()]


pos = {int(cleaned_data[i, 0]): (cleaned_data[i, 1], cleaned_data[i, 2]) for i in range(len(cleaned_data))}
nx.draw(G_both, pos, node_size=8, width=0.2, alpha=0.5, with_labels=False)
plt.title("Network plot")
#plt.show()
plt.savefig(f'Network plot.png')
plt.close('all')
nx.draw(G_both, pos, node_size=8, width=0, alpha=0.5, with_labels=False, node_color=node_colors)
plt.title("Cluster plot")
plt.savefig(f'Cluster plot.png')
#plt.show()
plt.close('all')

print("Combined graph nodes:", G_both.number_of_nodes())
print("Combined graph edges:", G_both.number_of_edges())
print("Isolated nodes:", len(list(nx.isolates(G_both))))
print("Amount of clusters:", len(clusters))
print("Cluster sizes:", [len(c) for c in clusters])


"""Iteration through layers"""
"""Explanation for myself/group: We currently have a graph with all branches (connections between couples of nodes) 
that satisfy both thresholds. For each layer, the iteration will check if that branch (between two nodes/fibres) satisfies
again both set thresholds to determine whether a fibre is clusterable throughout the full length."""

#Define constants
number_of_layers = 130
failure_fraction_allowed = 0.05
threshold_multiplier = 1.05
failure_limit = failure_fraction_allowed * number_of_layers
number_of_fibres = G_both.number_of_nodes()
clusters_updated = []

#Storage of removed fibers after iteration
remove_arr = set() 

thresholds = {}

previous_centroid = {"x": 0.0, "y": 0.0}

#Iterate through clusters
for clust in clusters:
    #Storage containers for fibers and clusters through layers
    fibre_counter = {}
    for fibre_id in clust:
        fibre_counter[fibre_id] = 0 

    cluster_fibre_id = clust

    sum_x = 0 
    sum_y = 0

    #Loop through fibres in current cluster (layer 0)
    for fibre_id in clust:
        fibre_row = cleaned_data[cleaned_data[:,0] == fibre_id][0]

        x = fibre_row[1]
        y = fibre_row[2]

        sum_x += x
        sum_y += y

    #Compute centroid for layer 0
    previous_centroid["x"] = sum_x / len(clust)
    previous_centroid["y"] = sum_y / len(clust)

    #Determine thresholds for layer 0
    for fibre_id in clust:
        fibre_row = cleaned_data[cleaned_data[:,0] == fibre_id][0]

        x = fibre_row[1]
        y = fibre_row[2]

        distance_centroid_0 = np.sqrt((x - previous_centroid["x"]) ** 2 + (y - previous_centroid["y"]) ** 2)

        #Threshold = distance + 5%
        thresholds[fibre_id] = (distance_centroid_0 * threshold_multiplier)

    #Iterate through layers
    for layer_idx, current_layer in enumerate(cleaned_data_i):

        for fibre_id in clust:

            #Find row belonging to this fibre
            fibre_row = current_layer[current_layer['fibre_id'] == fibre_id].iloc[0]

            x = fibre_row['x']
            y = fibre_row['y']

            #Distance to previous centroid
            distance = np.sqrt((x - previous_centroid["x"]) ** 2 + (y - previous_centroid["y"]) ** 2)

            #Compare against threshold
            if distance > thresholds[fibre_id]:
                fibre_counter[fibre_id] += 1

        #Update centroids to pass onto next layer
        sum_x = 0
        sum_y = 0

        for fibre_id in clust:

            fibre_row = current_layer[current_layer['fibre_id'] == fibre_id].iloc[0]

            x = fibre_row['x']
            y = fibre_row['y']

            sum_x += x
            sum_y += y

        previous_centroid["x"] = sum_x / len(clust)
        previous_centroid["y"] = sum_y / len(clust)

        #Update thresholds to pass onto next layer
        for fibre_id in clust:

            fibre_row = current_layer[current_layer['fibre_id'] == fibre_id].iloc[0]

            x = fibre_row['x']
            y = fibre_row['y']

            distance_i = np.sqrt((x - previous_centroid["x"]) ** 2 + (y - previous_centroid["y"]) ** 2)

            thresholds[fibre_id] = (distance_i * threshold_multiplier)

    current_remove = []

    for fibre_id in clust:

        if fibre_counter[fibre_id] > failure_limit:
            remove_arr.add(fibre_id)
            current_remove.append(fibre_id)

    #Created new updated clusters
    new_clust = []

    for fibre_id in clust:

        if fibre_id not in current_remove:
            new_clust.append(fibre_id)

    clusters_updated.append(new_clust)

#Add one final cluster containing all outliers
clusters_updated.append(list(remove_arr))

print(clusters_updated)
counter = 0
for kaas in clusters_updated:
    print(len(kaas))
    counter += 1
print(counter)
