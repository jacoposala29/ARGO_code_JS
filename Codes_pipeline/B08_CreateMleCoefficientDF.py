import h5py
import numpy as np
import pandas as pd
import pickle as pkl
from tools import create_ArgoProfileID
from TC_and_TS_define_param import grid_lower, grid_upper, grid_stride, year_start_TC, year_end_TC, folder2use

data_dir = folder2use + '/Outputs/'
results_dir = folder2use + '/Outputs/'
window_size_gp = 5

base_fn = (f'{results_dir}localMLESpaceTimeCoefs_Depth_DEPTH_8_20_{window_size_gp}'
           +f'_09_{year_start_TC}_{year_end_TC}_AllBasins.mat')

# Set dimensions of the grid
grid_start = int(grid_lower)
grid_end = int(grid_upper)
grid_stride = int(grid_stride)
DEPTH_IDX = int((grid_end - grid_start)/ grid_stride+1)

# Open .mat files from A12
# depth = 10
# fn_in = base_fn.replace('DEPTH', f'{depth:03d}')
# f = h5py.File(fn_in, 'r')
FILES = [h5py.File(base_fn.replace('DEPTH', f'{depth:03d}'), 'r')
         for depth in range(grid_start, grid_end+1, grid_stride)]

# Create Array of ArgoProfileIDs from the profiles in FILES
profile_ids = [
        create_ArgoProfileID(arr_fid[0], arr_cn[0])
        for arr_fid, arr_cn in zip(
            np.array(FILES[0]['FloatIDReg']),
            np.array(FILES[0]['CycleNumberReg']),
        )]
n_prof = len(profile_ids)

# Create mappings from ProfileID to Coefficients
VAR = [
    ('thetas',  'Phi'),
    ('thetat',  'Thetat'),
    ('sigma',   'Sigma'),
]

# Creates 3 files where each profile ID is associated with a coefficient from A12
for var, export_name in VAR:
    mat = np.zeros((n_prof, DEPTH_IDX))
    for idx in range(DEPTH_IDX):
        mat[:, idx] = np.array(FILES[idx][var]).flatten()
        dict_ = {
            k: v for k, v in zip(
                profile_ids,
                mat,
        )}
    pkl.dump(dict_, open(           # Save pkl files
        f'{results_dir}/Mle{export_name}Dict_{window_size_gp}.pkl', 'wb'))

# Open pkl files just created
phi_dict = pkl.load(open(
    f'{results_dir}/MlePhiDict_{window_size_gp}.pkl', 'rb'))
theta_t_dict = pkl.load(
        open(f'{results_dir}/MleThetatDict_{window_size_gp}.pkl', 'rb'))
sigma_dict = pkl.load(
        open(f'{results_dir}/MleSigmaDict_{window_size_gp}.pkl', 'rb'))

# Load B05 output
HU = pkl.load(open(f'{data_dir}/HurricaneAdjRawVariableDF.pkl', 'rb'))
# Create base DataFrame of unique before_pids (ID of profile 'before')
df = pd.DataFrame({
    'before_pid': HU['before_pid'].unique()
})

# Attach coefficient arrays for before_pids
df['phi'] = df['before_pid'].map(phi_dict)
df['theta_t'] = df['before_pid'].map(theta_t_dict)
df['sigma'] = df['before_pid'].map(sigma_dict)

# Create column to denote whether there is a nan at any row
phi_nan = df['phi'].apply(lambda x: np.sum(np.isnan(x)))
theta_t_nan = df['theta_t'].apply(lambda x: np.sum(np.isnan(x)))
sigma_nan = df['sigma'].apply(lambda x: np.sum(np.isnan(x))) #12221
np.sum(phi_nan>0)
np.sum(theta_t_nan>0)
np.sum(sigma_nan>0)
## No nan! :)

# Save dataframe
df.set_index('before_pid', inplace=True)
pkl.dump(df, open(f'{results_dir}/MleCoefDF_{window_size_gp}.pkl', 'wb')) 

