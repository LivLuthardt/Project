
import pandas as pd 
import numpy as np
from ourmain import df_cleaned
import matplotlib.pyplot as plt

def tangent_angles(df_cleaned):
    #consecutive point comparison
    df = df_cleaned.sort_values(['fibre_id', 'z']).copy()

    df['dx'] = df.groupby('fibre_id')['x'].diff()
    df['dy'] = df.groupby('fibre_id')['y'].diff()
    df['dz'] = df.groupby('fibre_id')['z'].diff()

    # zx and zy planar tilt
    df['angle_x_deg'] = np.degrees(np.arctan2(df['dx'], df['dz']))
    df['angle_y_deg'] = np.degrees(np.arctan2(df['dy'], df['dz']))
    
    # tilit angle 
    lateral_dist = np.sqrt(df['dx']**2 + df['dy']**2)
    df['tilt_angle_deg'] = np.degrees(np.arctan2(lateral_dist, df['dz']))
    
    # Return the dataframe with the new tilt columns, removing NaN: the first points
    return df.dropna(subset=['dx', 'dy', 'dz'])

df_final = tangent_angles(df_cleaned)

theta_tuples = list(zip(df_final['angle_x_deg'], df_final['angle_y_deg']))

theta_lists = []

for z in range(1, 129):
    subset = df_final[df_final['z'] == z]
    tuples = list(zip(subset['angle_x_deg'], subset['angle_y_deg']))
    theta_lists.append(tuples)

z_value = 11

z_specific_angles = theta_lists[z_value - 1]

angles = np.array(z_specific_angles)

theta_x = angles[:, 0]
theta_y = angles[:, 1]

corr = np.corrcoef(theta_x, theta_y)

print(corr)

plt.scatter(theta_x, theta_y, alpha=0.6)
plt.xlabel("Theta X (degrees)")
plt.ylabel("Theta Y (degrees)")
plt.title(f"Fibre angles at z = {z_value}")
plt.grid(True)
#plt.show()



#print(theta_tuples)
# Look for rows where tilt is noticeable
#tilted_samples = df_final[df_final['tilt_angle_deg'] > 5].head(10)
#print(tilted_samples[['fibre_id', 'z', 'angle_x_deg', 'angle_y_deg', 'tilt_angle_deg']])
#hello
# Look for rows where tilt is noticeable
#tilted_samples = df_final[df_final['tilt_angle_deg'] > 5].head(10)
#print(tilted_samples[['fibre_id', 'z', 'angle_x_deg', 'angle_y_deg', 'tilt_angle_deg']])
