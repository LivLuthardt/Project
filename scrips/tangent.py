
import pandas as pd 
import numpy as np
import plotly.express as px
from main import df_full

#tilt angle relative to the Z-axis
# sort.values to compare consecutive points
df = df_full.sort_values(['fibre_id', 'z'])

# Calculate differences between steps
df['dx'] = df.groupby('fibre_id')['x'].diff()
df['dy'] = df.groupby('fibre_id')['y'].diff()
df['dz'] = df.groupby('fibre_id')['z'].diff()

# angle = arctan(lateral_distance / vertical_distance)
# np.arctan2 handles division by zero apparently i cant use normal arctan
df['lateral_dist'] = np.sqrt(df['dx']**2 + df['dy']**2)
df['tilt_angle_rad'] = np.arctan2(df['lateral_dist'], df['dz'])
