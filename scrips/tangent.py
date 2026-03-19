
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
    fiber_summary = df.groupby('fibre_id').agg(
        # Mean columns
        x_mean=('x', 'mean'),
        y_mean=('y', 'mean'),
        angle_x_mean=('angle_x_deg', 'mean'),
        angle_y_mean=('angle_y_deg', 'mean'),
        tilt_angle_mean=('tilt_angle_deg', 'mean'),
        
        # Original/Non-mean versions (taking the first occurrence per fiber)
        x=('x', 'first'),
        y=('y', 'first'),
        angle_x_deg=('angle_x_deg', 'first'),
        angle_y_deg=('angle_y_deg', 'first'),
        tilt_angle_deg=('tilt_angle_deg', 'first')
    ).reset_index()

    n_fibers = int(len(df) / df['z'].nunique())
    
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
