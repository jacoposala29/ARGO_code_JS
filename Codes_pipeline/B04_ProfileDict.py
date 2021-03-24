# _____________________________________________________________________________
# This script creates a dictionary that contains Argo profiles' ID followed by 
# a 3-digit array from gridTempRes variable contained in the matlab file.
# Output: Hurricane{val}Dict.pkl
# _____________________________________________________________________________ 

# Minor differences with Addison code, but shouldn't have an impact


import h5py # Binary data format library. Opens matlab files 
from tools import create_ArgoProfileID # Import additional specific functions 
                                       # written by Donata in file tools.py
import scipy.io # Library 
import pickle as pkl # Library useful to write output files

# Import specific variables written by Donata in file TC_and_TS_define_param.py
from TC_and_TS_define_param import (
        window_size,min_num_obs, # Returns: 8, 20
        var2use, # Returns: Temperature
        folder2use,
        )

for fi, val in [('ProfFiltered', 'Prof'), ('Res', 'Res')]:
    
    mat = h5py.File(str(folder2use) + f'/Outputs/'
        f'gridArgo{fi}_'+window_size+'_'+min_num_obs+'_'+var2use+'.mat','r')

    # Create a dictionary 
    dict_ = {
            create_ArgoProfileID(arr_fid[0], arr_cn[0]) : arr_rid
            for arr_fid, arr_cn, arr_rid
            in zip(mat['profFloatIDAggrSel'][()], # Assign 
                                                    # mat['profFloatIDAggrSel'].value 
                                                    # to arr_fid 
                   mat['profCycleNumberAggrSel'][()], # Assign 
                                                        # mat['profCycleNumberAggrSel'].value 
                                                        # to arr_cn
                   mat[f'gridVarObs{val}'][()].T, # Assign mat[''gridVarObs{val}'].value
                                                  # to arr_rid. transposed matrix.
                                                  # val is Res
                  )
            }

    pkl.dump(dict_, open(str(folder2use) + f'/Outputs/Hurricane{val}Dict.pkl', 'wb')) #Output file
    
    
    