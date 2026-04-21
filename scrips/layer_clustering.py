import pandas as pd 
import numpy as np
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from main import data_clean
import matplotlib as plt

layer0 = data_clean[data_clean['z'] == 0]
layer0 = layer0.reset_index(drop=True)

features = ['fibre_id', 'x', 'y', 'angle_x_deg', 'angle_y_deg']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(layer0[features])

X= data_clean['z'].to_numpy()

# finding neighbors 
nbrs = NearestNeighbors(n_neighbors=100, algorithm='auto').fit(X)

clust = KNeighborsClassifier(n_neighbors=5, weights='uniform')

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
