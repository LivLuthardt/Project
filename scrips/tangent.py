
import pandas as pd 
import numpy as np
from main import df_full

def get_tangent_angles(df_cleaned):
    # Use your exact sorting logic to ensure consecutive point comparison
    df = df_cleaned.sort_values(['fibre_id', 'z']).copy()

    # Calculate differences (Finite-Difference Method) [cite: 102]
    df['dx'] = df.groupby('fibre_id')['x'].diff()
    df['dy'] = df.groupby('fibre_id')['y'].diff()
    df['dz'] = df.groupby('fibre_id')['z'].diff()

    # Calculate and store the Planar Tilts (X and Y rotations)
    # These represent the ZX and ZY tilts mentioned in your objectives [cite: 100]
    df['angle_x_deg'] = np.degrees(np.arctan2(df['dx'], df['dz']))
    df['angle_y_deg'] = np.degrees(np.arctan2(df['dy'], df['dz']))
    
    # Calculate Total Misalignment Magnitude (the 'phi' angle)
    lateral_dist = np.sqrt(df['dx']**2 + df['dy']**2)
    df['tilt_angle_deg'] = np.degrees(np.arctan2(lateral_dist, df['dz']))
    
    # Return the dataframe with the new tilt columns, removing NaN rows from .diff()
    return df.dropna(subset=['dx', 'dy', 'dz'])

# Now you call it using df_full
df_final = get_tangent_angles(df_cleaned)