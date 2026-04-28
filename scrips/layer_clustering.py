import pandas as pd 
import numpy as np
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

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
    neighbours = []
    for pindex in range(len(points)):
        neighbours.append(tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][pindex]:tri.vertex_neighbor_vertices[0][pindex+1]])

    print(neighbours)
    plt.figure() 
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o', markersize=2)
    plt.show()

    
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

