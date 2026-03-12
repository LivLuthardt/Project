
import pandas as pd 
import numpy as np

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
    df.dropna(subset=['dx', 'dy', 'dz'])

    return df

def fiber_summary(df):
    # Create the fiber summary with mean values
    fiber_summary = df.groupby('fibre_id').agg({
        'x': 'mean',
        'y': 'mean',
        'angle_x_deg': 'mean',
        'angle_y_deg': 'mean',
        'tilt_angle_deg': 'mean'
    }).reset_index()

    # Rename the columns to reflect they are means
    fiber_summary.columns = ['fibre_id', 'x_mean', 'y_mean', 'angle_x_mean', 'angle_y_mean', 'tilt_angle_mean']
    return fiber_summary

