import pandas as pd 
import numpy as np
import scipy
from sklearn import neighbors
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy import spatial 



# layer0 = data_clean[data_clean['z'] == 0]
# layer0 = layer0.reset_index(drop=True)

# features = ['fibre_id', 'x', 'y', 'angle_x_deg', 'angle_y_deg']
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(layer0[features])

# X= data_clean['z'].to_numpy()

#     # finding neighbors 
# nbrs = NearestNeighbors(n_neighbors=100, algorithm='auto').fit(X)

# clust = KNeighborsClassifier(n_neighbors=5, weights='uniform')

# #Y = [[1], [0], [3], [77], [2], [9]]
# #A = kneighbors_graph(Y, 2, mode='connectivity', include_self=True)
# #A.toarray()
# #print(A)
# all_z = sorted(data_clean['z'].unique())

# for z in all_z[1,:]:

#     layer = data_clean[data_clean['z'] == z].reset_index(drop=True)

#---------------------------------Delauney-------------------------------------------------
def delaunay_triangulation(df):

    subset = df[df['z_idx'] == 1]
    points = subset[['x', 'y']].to_numpy()
    tri = Delaunay(points)
    
    plt.figure() 
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o', markersize=2)
    plt.show()


'''
import scipy.sparse as sp

# 2. Extract edges from triangles
edges = set()
for simplex in tri.simplices:
    for i in range(3):
        # Sort to ensure (u, v) is same as (v, u)
        edge = tuple(sorted((simplex[i], simplex[(i+1)%3])))                
        edges.add(edge)

# 3. Find Nearest Neighbor for point 0 using edges
p_idx = 0
min_dist = float('inf')
closest_neighbor = -1

for edge in edges:
    if p_idx in edge:
        # Get the neighbor index
        neighbor_idx = edge[0] if edge[1] == p_idx else edge[1]
        
        # Calculate Euclidean distance
        dist = np.linalg.norm(points[p_idx] - points[neighbor_idx])
        
        if dist < min_dist:
            min_dist = dist
            closest_neighbor = neighbor_idx

print(f"Nearest neighbor to point {p_idx} is {closest_neighbor} at distance {min_dist}")
'''

#---------------------------------Stef's try to do this stuff------------------------------
def eqdqzd(df):
    points = ['x','y']
    subset = df.loc[df['z'] == 0, points]

    nbrs = NearestNeighbors(n_neighbors=2).fit(subset)
    distances,ids = nbrs.kneighbors(subset)

    distances = distances[:,1] # Distance to nearest neighbor (excluding self)
    print(distances)

    distances_sorted = sorted(distances)
    plt.plot(distances_sorted)
    plt.xlabel('Points sorted by distance to nearest neighbor')
    plt.ylabel('Distance to nearest neighbor')
    plt.title('K-distance Graph for eps selection')
    plt.show()

