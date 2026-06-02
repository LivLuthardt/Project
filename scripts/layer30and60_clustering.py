import pandas as pd 
import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from data_clean import data_cleaned
from tangent import tangent_angles_central
import matplotlib.pyplot as plt
from kneed import KneeLocator
import matplotlib.patches as mpatches
from plot import plot_fibers_clustered

""" ------------------------------------------- Import data and manipulate dataframe ------------------------------------------- """
#Import data and clean it
raw_df = pd.read_csv('raw_data.csv')    #Original data
syn_df = pd.read_csv('sim_data.csv')    #Synthetic data

data_clean = data_cleaned(raw_df)       #Original data
#No cleaned data for synthetic fibers

df = tangent_angles_central(data_clean) #Original data

layer = 60
dataset = 'sythetic' #either put 'original' if df is used or 'synthetic' if syn_df is used

#Define layer_0 data
layer_0 = syn_df[syn_df['z_idx'] == layer] #change the dataset depending on if you want to use the original or synthetic dataset
layer_0 = layer_0.reset_index(drop=True)
#Consider only required features for clustering
features = ['fibre_id', 'x', 'y']
cleaned_data = layer_0[['x', 'y']]
cleaned_data = np.column_stack((layer_0['fibre_id'].values, cleaned_data))


""" --------- Neighborhood rule functions, store cartesian distances and respective angles between each pair of fibers --------- """
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

""" ---------------------- Plot histogram of distances and angles, and determine threshold as percentiles ---------------------- """
#Distance
scores_0_d = []
layer_0_results_d = good_neighbor_distance(cleaned_data)
for item in layer_0_results_d:
    scores_0_d.append(item[2])

#Choose threshold percentile
pct_d = 95

threshold_distance = np.percentile(scores_0_d, pct_d)

# Plot histogram
plt.figure()
plt.hist(scores_0_d, bins=100)
plt.axvline(threshold_distance,color = 'r', label = 'Threshold') 
plt.title("Distance Histogram")
plt.xlabel("Distance between pairs of points")
plt.ylabel("Frequency")
plt.legend()
plt.savefig(fname = f'Images/Layer {layer} {dataset} distance histogram')
plt.close('all')
print("Threshold_Distance", threshold_distance)


""" ------------------------------------- Build distance and angle matrix and create graph ------------------------------------- """
results_distance = good_neighbor_distance(cleaned_data)

#Map fibre_id to row index
fibre_ids = cleaned_data[:, 0].astype(int)
id_to_idx = {fid: i for i, fid in enumerate(fibre_ids)}

#Build distance and angle matrices
n_d = len(fibre_ids)
D_d = np.full((n_d, n_d), np.inf)
np.fill_diagonal(D_d, 0)
"""
n_a = len(fibre_ids)
D_a = np.full((n_a, n_a), np.inf)
np.fill_diagonal(D_a, 0)
"""
#Fill matrices
for fibre_d_i, fibre_d_j, score in results_distance:
    i = id_to_idx[int(fibre_d_i)]
    j = id_to_idx[int(fibre_d_j)]
    D_d[i, j] = score
    D_d[j, i] = score


""""Build combined graph"""
G_both = nx.Graph()
G_both.add_nodes_from([int(fid) for fid in fibre_ids])

#Build graph directly from thresholds
#Maximum physical interaction radius
max_radius = 42  #7*6 6 circles around one fiber

for i in range(len(fibre_ids)):
    fid_i = int(fibre_ids[i])

    x_i = cleaned_data[i, 1]
    y_i = cleaned_data[i, 2]

    for j in range(i + 1, len(fibre_ids)):
        fid_j = int(fibre_ids[j])
        x_j = cleaned_data[j, 1]
        y_j = cleaned_data[j, 2]

        #Physical Euclidean distance
        euclidean_distance = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)

        #Skip distant fibres
        if euclidean_distance > max_radius:
            continue

        score_d = D_d[i, j]
        #score_a = D_a[i, j]

        #Apply thresholds separately
        if (score_d <= threshold_distance): #and score_a <= threshold_angle):
            combined_score = score_d #+ score_a
            similarity = 1 / (combined_score + 1e-12)
            G_both.add_edge(fid_i, fid_j, weight=similarity)

print("Combined graph nodes:", G_both.number_of_nodes())
print("Combined graph edges:", G_both.number_of_edges())
print("Isolated nodes:", len(list(nx.isolates(G_both))))

#Create list for isolated nodes to store in 3D outlier cluster
isolated_nodes = []
for isol in nx.isolates(G_both):
    isolated_nodes.append(isol)
print(isolated_nodes)

G_cluster = G_both.copy()

#Remove isolated fibres
G_cluster.remove_nodes_from(list(nx.isolates(G_cluster)))

#Creates clusters based on densely populated nodes
communities = nx.community.greedy_modularity_communities(G_cluster, weight="weight", resolution=15, cutoff=10)
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


""" ---------------------------------------- Assign colors to clusters and show graphs ---------------------------------------- """
#Assign colors
num_clusters = max(node_to_cluster.values()) + 1
colors = plt.cm.tab20(np.linspace(0, 1, num_clusters + 1))

#Last color reserved for unclustered nodes
default_cluster = num_clusters

#Create node colors safely
node_colors = [colors[node_to_cluster.get(node, default_cluster)] for node in G_both.nodes()]

pos = {int(cleaned_data[i, 0]): (cleaned_data[i, 1], cleaned_data[i, 2]) for i in range(len(cleaned_data))}
nx.draw(G_both, pos, node_size=8, width=0.2, alpha=0.5, with_labels=False)
plt.title("Network plot")
plt.axis('equal')
#plt.show()
plt.savefig(f'Images/Layer {layer} {dataset} network plot.png')
plt.close('all')
nx.draw(G_both, pos, node_size=8, width=0, alpha=0.5, with_labels=False, node_color=node_colors)
plt.title("Cluster plot")
plt.axis('equal')
plt.savefig(f'Images/Layer {layer} {dataset} cluster plot.png')
#plt.show()
plt.close('all')
 
print("Combined graph nodes:", G_both.number_of_nodes())
print("Combined graph edges:", G_both.number_of_edges())
print("Isolated nodes:", len(list(nx.isolates(G_both))))
print("Amount of clusters:", len(clusters))
print("Cluster sizes:", [len(c) for c in clusters])