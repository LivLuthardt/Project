
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
    
    # Return the dataframe with the new tilt columns
    return df

def fiber_summary(df):
    # Use named aggregation to define names and functions simultaneously
    means = df.groupby('fibre_id', 'z').agg(
        x_mean=('x', 'mean'),
        y_mean=('y', 'mean'),
        angle_x_mean=('angle_x_deg', 'mean'),
        angle_y_mean=('angle_y_deg', 'mean'),
        tilt_angle_mean=('tilt_angle_deg', 'mean'),
    ).reset_index()

    summary_df = df.merge(means, on='fibre_id', how='left')

    n_fibers = df['fibre_id'].nunique()
    
    return fiber_summary, n_fibers

def tangent_angles_central(df_cleaned):
    df = df_cleaned.sort_values(['fibre_id', 'z']).copy()

    x_prev = df.groupby('fibre_id')['x'].shift(1)
    x_next = df.groupby('fibre_id')['x'].shift(-1)

    y_prev = df.groupby('fibre_id')['y'].shift(1)
    y_next = df.groupby('fibre_id')['y'].shift(-1)

    z_prev = df.groupby('fibre_id')['z'].shift(1)
    z_next = df.groupby('fibre_id')['z'].shift(-1)

    df['dx'] = x_next - x_prev
    df['dy'] = y_next - y_prev
    df['dz'] = z_next - z_prev

    df['angle_x_deg'] = np.degrees(np.arctan2(df['dx'], df['dz']))
    df['angle_y_deg'] = np.degrees(np.arctan2(df['dy'], df['dz']))

    lateral_dist = np.sqrt(df['dx']**2 + df['dy']**2)
    df['tilt_angle_deg'] = np.degrees(np.arctan2(lateral_dist, df['dz']))

    return df
