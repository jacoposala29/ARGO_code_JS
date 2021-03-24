# _____________________________________________________________________________
# This script concatenates Tc tracks from the 5 basins into a single dataframe. 
# Then it calculates the distance of Argo profiles from the TC tracks.
# It saves Argo profiles that are near TC tracks and not near Tc. 
# _____________________________________________________________________________

# Import libraries needed for the program to run
import numpy as np   # Library to compute most math operations
import pandas as pd  # Library to handle dataframes - super useful
import pickle as pkl # Library useful to write output files

# Import specific variables written by Donata in file TC_and_TS_define_param.py
from TC_and_TS_define_param import (
        AL,EP,WP,SH,IO, # TC tracks in those 5 basins
        year_start_TC,  # 2007
        year_end_TC, # 2010
        folder2use
        )

# Concatenate TC tracks from the 5 basins into a single pandas dataframe
# Note: tracks data are available every 6 hours
Hurricanes = pd.concat([AL, EP, WP, SH, IO])
# Select only TC on or after 2007
Hurricanes = Hurricanes[Hurricanes['SEASON'] >= year_start_TC]
# Select only TC on or before 2010
Hurricanes = Hurricanes[Hurricanes['SEASON'] <= year_end_TC]
# Reset pandas native index
Hurricanes = Hurricanes.reset_index(drop=True)
# Convert date time to pandas datetime format
Hurricanes['Timestamp_PD'] = Hurricanes['TIMESTAMP'].apply(
        lambda x: pd.Timestamp(x))

# Load metadata of Argo profiles from 2007 to 2010 as output of B00 script
Profiles = pkl.load(open(str(folder2use) + '/Outputs/ArgoProfileDF.pkl', 'rb'))

# Create empty pandas Series to store Argo profiles near TC tracks
near_hur = pd.Series(np.repeat(0, Profiles.shape[0]), dtype=int)

# Loop through all considered TC track time stamps (every 6 hour for every track)
for idx, row in Hurricanes.iterrows():
    # Print on screen iteration number (every 1,000th iteration)
    if (idx % 1000 == 0):
        print(f'Now processing {idx}')
    
    # Calculate distance of all Argo profiles from current TC track 
    dist = (  (Profiles['Latitude']  - row['LAT'])  ** 2
            + (Profiles['Longitude'] - row['LONG']) ** 2).values
    # Calculate temporal separation of all Argo profiles from current TC track
    # Negative if Argo profile was earlier than current TC track
    tdiff = (Profiles['Timestamp'] - row['Timestamp_PD'])
    
    # Determine Argo profiles that are close (in space and time) to current TC track
    # near_hur is a metric which at each iteration can give True 
    # (1, if Argo was close to a TC track) or False (0, if Argo was NOT close to a TC track)
    # At each iteration of the for loop, the new near_hur values are summed to
    # the total from the previous iteration, and so on, so that at the end of the loop,
    # near_hur will be equal to the number of TC track time stamps the Argo profile was close to
    near_hur += pd.Series((                            # gives True if Argo profile was...
            (tdiff >= pd.Timedelta(days=-30)).values * # not earlier than 30 days before TC track
            (tdiff <= pd.Timedelta(days=2)).values   * # AND not later than 2 days after TC track
            (dist <= 64)                               # AND close in distance 
            ), dtype=int)

# Save Argo profiles which are near a TC track ? Right now it saves all of them...
# Add column to Profiles dataframe to specify near_hur metric
Profiles['NearHurricane'] = near_hur
# Sort rows based on date and time
df_sorted = Profiles.sort_values('Timestamp', ascending=False)
# Drop duplicates of Profile ID if there are any
df = df_sorted.drop_duplicates('ProfileID', keep='first')
# Reset pandas index
df = df.reset_index(drop=True)
# Write data to file
pkl.dump(df, open(str(folder2use) + '/Outputs/ArgoProfileDF_NearHur.pkl', 'wb'))


#______________________________________________________________________________
# THIS WAS CHANGED BY JACOPO
# Drop duplicates of Profile ID if there are any
df = df_sorted.drop_duplicates('ProfileID', keep='first')
# Reset pandas index
df = df.reset_index(drop=True)
# Save Argo profiles which are near a TC track
df_hur = df[df['NearHurricane'] != 0]
# Reset pandas index
df = df_hur.reset_index(drop=True)
# Write data to file
pkl.dump(df, open(str(folder2use) + '/Outputs/ArgoProfileDF_NearHur_J.pkl', 'wb'))

#______________________________________________________________________________

# Drop duplicates of Profile ID if there are any
df = df_sorted.drop_duplicates('ProfileID', keep='first')
# Reset pandas index
df = df.reset_index(drop=True)
# Save Argo profiles which are NOT near a TC track
df_no_hur = df[df['NearHurricane'] == 0]
# Reset pandas index
df = df_no_hur.reset_index(drop=True)
# Write data to file
pkl.dump(df, open(str(folder2use) + '/Outputs/ArgoProfileDF_NoHur.pkl', 'wb'))


