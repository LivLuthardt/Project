import pandas as pd 
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from main import data_clean

layer0 = data_clean[data_clean['z'] == 0]
layer0 = layer0.reset_index(drop=True)

features = ['x', 'y', 'angle_x_deg', 'angle_y_deg']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(layer0[features])

X= data_clean['z'].to_numpy()

# finding neighbors 
nbrs = NearestNeighbors(n_neighbors=100, algorithm='auto').fit(X)