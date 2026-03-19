import pandas as pd 
import numpy as np
import plotly.express as px

def data_cleaned(df):
    #PART 1: CLEANING NAN
    #Data Import
    df = pd.read_csv('raw_data.csv')
    #Data Cleanup
    df_clean = df.dropna()

    #3D plot
    ''' 
    fig = px.line_3d(df, x="x", y="y", z="z", color='fibre_id')
    fig.update_layout(scene=dict(aspectmode='manual',aspectratio=dict(x=15,y=7.5,z=1)))
    fig.show()
    '''
    #PART 2: CLEANING SHORT ONES
    # remove ones that dont reach the full z length 
    # global end‑points
    zmin = df_clean['z'].min()
    zmax = df_clean['z'].max()

    # take z values for each fibre, compute min and max, return new DataFrame indexed by fibre_id
    #agg: computes BOTH min and max
    z_ext = df_clean.groupby('fibre_id')['z'].agg(['min','max'])

    # strict test for full_ids
    full_ids = z_ext[(z_ext['min'] == zmin) & (z_ext['max'] == zmax)].index
    partial_ids = z_ext.index.difference(full_ids)

    # make two dataframes
    df_full = df_clean[df_clean['fibre_id'].isin(full_ids)]
    df_partial = df_clean[df_clean['fibre_id'].isin(partial_ids)]

    # print(f"{len(full_ids)} fibres span [{zmin},{zmax}]; "
    #     f"{len(partial_ids)} do not.")

    # plot both sets
    '''
    for dset, title in ((df_full,    "Fibres reaching full z range"),
                        (df_partial, "Fibres not reaching full z range")):
        fig = px.line_3d(dset,
                        x="x", y="y", z="z",
                        color="fibre_id",
                        title=title)
        fig.update_layout(
            scene=dict(aspectmode="manual",
                    aspectratio=dict(x=15, y=7.5, z=1))
        )
        fig.show()
    '''
    #PART 3: CLEANING WONKY ONES
    # tilt angle relative to the Z-axis
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

    # find maximum kink for every fiber
    max_tilts = df.groupby('fibre_id')['tilt_angle_rad'].max()
    Q1 = max_tilts.quantile(0.25)
    Q3 = max_tilts.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    import pandas as pd 
import numpy as np
import plotly.express as px

def data_cleaned(df):
    #PART 1: CLEANING NAN
    #Data Import
    df = pd.read_csv('raw_data.csv')
    #Data Cleanup
    df_clean = df.dropna()

    #3D plot
    ''' 
    fig = px.line_3d(df, x="x", y="y", z="z", color='fibre_id')
    fig.update_layout(scene=dict(aspectmode='manual',aspectratio=dict(x=15,y=7.5,z=1)))
    fig.show()
    '''
    #PART 2: CLEANING SHORT ONES
    # remove ones that dont reach the full z length 
    # global end‑points
    zmin = df_clean['z'].min()
    zmax = df_clean['z'].max()

    # take z values for each fibre, compute min and max, return new DataFrame indexed by fibre_id
    #agg: computes BOTH min and max
    z_ext = df_clean.groupby('fibre_id')['z'].agg(['min','max'])

    # strict test for full_ids
    full_ids = z_ext[(z_ext['min'] == zmin) & (z_ext['max'] == zmax)].index
    partial_ids = z_ext.index.difference(full_ids)

    # make two dataframes
    df_full = df_clean[df_clean['fibre_id'].isin(full_ids)]
    df_partial = df_clean[df_clean['fibre_id'].isin(partial_ids)]

    # print(f"{len(full_ids)} fibres span [{zmin},{zmax}]; "
    #     f"{len(partial_ids)} do not.")

    # plot both sets
    '''
    for dset, title in ((df_full,    "Fibres reaching full z range"),
                        (df_partial, "Fibres not reaching full z range")):
        fig = px.line_3d(dset,
                        x="x", y="y", z="z",
                        color="fibre_id",
                        title=title)
        fig.update_layout(
            scene=dict(aspectmode="manual",
                    aspectratio=dict(x=15, y=7.5, z=1))
        )
        fig.show()
    '''
    #PART 3: CLEANING WONKY ONES
    # tilt angle relative to the Z-axis
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

    # find maximum kink for every fiber
    max_tilts = df.groupby('fibre_id')['tilt_angle_rad'].max()
    Q1 = max_tilts.quantile(0.25)
    Q3 = max_tilts.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

# Separate IDs based on the IQR bounds
    kinked_ids = max_tilts[(max_tilts > upper_bound) | (max_tilts < lower_bound)].index
    clean_ids = max_tilts[(max_tilts <= upper_bound) & (max_tilts >= lower_bound)].index

# Filter the original dataframe by ID
    df_kinked = df[df['fibre_id'].isin(kinked_ids)]
    df_cleaned = df[df['fibre_id'].isin(clean_ids)]

    print(f" - {len(clean_ids)} fibres kept.")
    print(f" - {len(kinked_ids)} fibres removed")

    return df_cleaned

# Separate IDs based on the IQR bounds
    kinked_ids = max_tilts[(max_tilts > upper_bound) & (max_tilts < lower_bound)].index
    clean_ids = max_tilts[(max_tilts <= upper_bound) & (max_tilts >= lower_bound)].index

# Filter the original dataframe by ID
    df_kinked = df[df['fibre_id'].isin(kinked_ids)]
    df_cleaned = df[df['fibre_id'].isin(clean_ids)]

    print(f" - {len(clean_ids)} fibres kept.")
    print(f" - {len(kinked_ids)} fibres removed")

    return df_cleaned