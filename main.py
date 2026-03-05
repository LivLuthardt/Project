#Data Import
import pandas as pd 

df = pd.read_csv('raw_data.csv')
#Data Cleanup
df_clean = df.dropna()
#3D Plot
import plotly.express as px

fig = px.line_3d(df, x="x", y="y", z="z", color='fibre_id')
fig.update_layout(scene=dict(aspectmode='manual',aspectratio=dict(x=15,y=7.5,z=1)))
fig.show()
#getting rid of short ones 
'''import numpy as np 

def fibre_length(group):
    # group is a DataFrame containing all rows for one fibre
    # difference between rows
    diff = group[['x', 'y', 'z']].diff().dropna()
    dist = np.sqrt((diff**2).sum(axis=1))
    return dist.sum()

# appply to each fibre
#grouping the cleaned data
fibre_lengths = df_clean.groupby('fibre_id').apply(fibre_length, include_groups=False)

#threshold (used range of z for this)
z_total = df_clean['z'].max() - df_clean['z'].min()
threshold = 0.5*z_total 
#take their fibre ids (index)
valid_fibres = fibre_lengths[fibre_lengths >= threshold].index

#True for rows that are in a valid fibre (tells which ones should not be cut)
df_filtered = df_clean['fibre_id'].isin(valid_fibres)
# cleaned-up data
df_trimmed = df_clean[df_filtered]  

print(f"Removed {len(fibre_lengths) - len(valid_fibres)} short fibers.")'''
#plotting short ones
''' 
import plotly.express as px

# Identify removed fibers
removed_ids = fibre_lengths[fibre_lengths < threshold].index
#for every row in df_clean: True when the row's fibre id is in removed_ids list
#df_removed only contains removed points 
df_removed = df_clean[df_clean['fibre_id'].isin(removed_ids)]

fig = px.line_3d(
    df_removed, 
    x="x", 
    y="y", 
    z="z", 
    color='fibre_id', 
    line_group='fibre_id', 
    title="Removed Short Fibers"
)

# Your aspect ratio settings
fig.update_layout(
    scene=dict(
        aspectmode='manual', 
        aspectratio=dict(x=15, y=7.5, z=1)
    )
)

fig.show()
'''

import numpy as np
import pandas as pd
import plotly.express as px

# tilt angle relative to the Z-axis
# sort.values to compare consecutive points
df = df_clean.sort_values(['fibre_id', 'z'])

# Calculate differences between steps
df['dx'] = df.groupby('fibre_id')['x'].diff()
df['dy'] = df.groupby('fibre_id')['y'].diff()
df['dz'] = df.groupby('fibre_id')['z'].diff()

# angle = arctan(lateral_distance / vertical_distance)
# np.arctan2 handles division by zero apparently i cant use normal arctan
df['lateral_dist'] = np.sqrt(df['dx']**2 + df['dy']**2)
df['tilt_angle_rad'] = np.arctan2(df['lateral_dist'], df['dz'])

# find maximum kink for every fiber
threshold = 45
df['tilt_angle_deg'] = np.degrees(df['tilt_angle_rad'])
max_tilts = df.groupby('fibre_id')['tilt_angle_deg'].max()

kinked_ids = max_tilts[max_tilts > threshold].index
clean_ids = max_tilts[max_tilts <= threshold].index

df_kinked = df[df['fibre_id'].isin(kinked_ids)]
df_cleaned = df[df['fibre_id'].isin(clean_ids)]

print(f"Analysis complete:")
print(f" - {len(kinked_ids)} fibres removed (Max tilt > {threshold}°)")
print(f" - {len(clean_ids)} fibres kept.")

#plot it
fig_clean = px.line_3d(df_clean, x="x", y="y", z="z", color="fibre_id", title="Cleaned")
fig_clean.update_layout(scene=dict(aspectratio=dict(x=15, y=7.5, z=1)))
fig_clean.show()

fig_kink = px.line_3d(df_kinked, x="x", y="y", z="z", color="fibre_id", title="Removed Kinks")
fig_kink.update_layout(scene=dict(aspectratio=dict(x=15, y=7.5, z=1)))
fig_kink.show()