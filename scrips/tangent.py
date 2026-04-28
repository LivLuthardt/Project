import pandas as pd 
import numpy as np

def tangent_angles_central(df_cleaned):
    # Using central difference method: (next - prev)
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

    cols_to_fill = ['angle_x_deg', 'angle_y_deg', 'tilt_angle_deg']
    df[cols_to_fill] = df.groupby('fibre_id')[cols_to_fill].bfill().ffill()

    return df

def fiber_summary(df):
    # Apply central difference calculation
    df_angles = tangent_angles_central(df)
    
    # Calculate means per fiber
    # Note: Removed 'z' from groupby so we get one row per fiber_id
    means = df_angles.groupby('fibre_id').agg(
        x_mean=('x', 'mean'),
        y_mean=('y', 'mean'),
        angle_x_mean=('angle_x_deg', 'mean'),
        angle_y_mean=('angle_y_deg', 'mean'),
        tilt_angle_mean=('tilt_angle_deg', 'mean'),
    ).reset_index()

    # Merge means back to the original dataframe to keep all columns (including z)
    summary_df = df_angles.merge(means, on='fibre_id', how='left')

    n_fibers = df['fibre_id'].nunique()
    
    return summary_df, n_fibers

from scipy.stats import ks_2samp


def ks_global(df):
    # Drop NaNs to avoid issues
    x_finite = df["angle_x_deg"].dropna()
    x_ellipse = df["EllipseXTilt"].dropna()

    y_finite = df["angle_y_deg"].dropna()
    y_ellipse = df["EllipseYTilt"].dropna()

    ks_x = ks_2samp(x_finite, x_ellipse)
    ks_y = ks_2samp(y_finite, y_ellipse)

    return ks_x, ks_y
