from sklearn.neighbors import NearestNeighbors
import numpy as np

def neighbors(df):
    points = ['x','y']
    nbrs = NearestNeighbors(n_neighbors=2).fit(df.loc[df['z'] == 0, points])
    distances,_ = nbrs.kneighbors(df.loc[df['z'] == 0, points])
    distances = distances[:,1] #Gets the distance between point and nearest neighbor
    print(distances.shape)