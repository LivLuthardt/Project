import pandas as pd 
import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from data_clean import data_cleaned
from tangent import tangent_angles_central
import matplotlib.pyplot as plt

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
            weight_d = 3
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
n_std = 1
threshold = mean_scores_0 + n_std * std_scores_0
# Plot histogram
plt.hist(scores_0, bins=100)
plt.axvline(threshold)  
plt.show()
print("Mean:", mean_scores_0)
print("Std:", std_scores_0)
print("Threshold", threshold)

   
"""Initialise knn on the initial layer (layer 0) and plot the graph"""
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