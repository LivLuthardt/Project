import pandas as pd 
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from main import data_clean

layer0 = data_clean[data_clean['z'] == 0]
layer0 = layer0.reset_index(drop=True)

features = ['x', 'y', 'angle_x_deg', 'angle_y_deg']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(layer0[features])