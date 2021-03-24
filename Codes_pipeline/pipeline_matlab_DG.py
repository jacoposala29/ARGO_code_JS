# Import libraries needed for the program to run
from datetime import datetime # Library that includes datetime in pandas format
import os #module that determines which module to load for path based on which operating system you have
import subprocess #module that runs new applications or programs by creating new processes  
import sys #module that gives us information about constants, functions, and methods of the interpreter 
import echo 

# Import additional specific functions written by Donata in file tools.py
from tools import replace 
from itertools import product 

# Import specific functions written by Donata in file TC_and_TS_define_param.py
from TC_and_TS_define_param import (
        year_pairs, # Returns ((2007, 2008), (2009, 2010))
        year_start_TC,year_end_TC, # Start: 2007 End: 2010
        matlab_path, # matlab directory
        grid_lower,grid_upper,grid_stride, #lower: 10, upper: 210, stride: 100
        window_size,min_num_obs, # Returns 8, 20
        var2use,center_month,OB, # Temperature, 9, Oceanic basins
        windowSizeGP,n_parpool, # 5, 1
        depth_layers, #3
        folder2use,
        case2use,
        )

# List of Matlab programs to run
list2run = str(sys.argv)#['A05', 'A06','A07','A08','A09','A10','A11','A12'] #['A01', 'A02', 'A03', 'A04'] #

# Define a function to run within python a script that was written in Matlab languge
def matlab_run(script: str, replacement_dict: dict = {}):
    timestamp = datetime.now()
    f_out = f"temp_matlab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.m"
    replace(script, f_out, replacement_dict)
    proc=subprocess.run([
        matlab_path,
        '-nodisplay',
        '-nosplash',
        '-nodesktop',
        f'-r "run(\'{f_out}\');exit;"'
        ])
    if proc.returncode != 0:
        raise RuntimeError(f'Subprocess {f_out} exited with non-zero return '
                'status.')
    # cleanup
    os.remove(f_out)
    return

## A01: gridding, saves (temperature) observations from Argo profiles on 3 different pressure depths
if ('A01' in list2run):
    print('>>>>>>>>>>>>>>>>>>>>>>>>A01')
    if case2use == 'ML':
        YEARS_LIST = year_pairs
        for YEARS in YEARS_LIST:
            matlab_run('A01_MixedLayer.m', {
                '<PY:YEARS>': f'{YEARS[0]}_{YEARS[1]}',
                '<PY:DATA_LOC>': '../Inputs/',
                '<PY:GRID_LOWER>': grid_lower,
                '<PY:GRID_UPPER>': grid_upper,
                '<PY:GRID_STRIDE>': grid_stride,
                '<PY:VAR2USE>': var2use,
                '<PY:FOLDER2USE>': folder2use,
                })       
    else:      
        YEARS_LIST = year_pairs
        for YEARS in YEARS_LIST:
            matlab_run('A01_pchipGridding.m', {
                '<PY:YEARS>': f'{YEARS[0]}_{YEARS[1]}',
                '<PY:DATA_LOC>': '../Inputs/',
                '<PY:GRID_LOWER>': grid_lower,
                '<PY:GRID_UPPER>': grid_upper,
                '<PY:GRID_STRIDE>': grid_stride,
                '<PY:VAR2USE>': var2use,
                '<PY:FOLDER2USE>': folder2use,
                })

## A02: concatenates arrays
if ('A02' in list2run):
    print('>>>>>>>>>>>>>>>>>>>>>>>>A02')

    matlab_run('A02_concatenateArrays.m', {
            '<PY:YEARS>': f'{year_pairs}',
            '<PY:VAR2USE>': var2use,
            '<PY:FOLDER2USE>': folder2use,            
            })

## A03: creates data mask
if ('A03' in list2run):
    print('>>>>>>>>>>>>>>>>>>>>>>>>A03')

    matlab_run('A03_createDataMask.m', {
           '<PY:WINDOW_SIZE>': window_size,
           '<PY:MIN_NUM_OBS>': min_num_obs,
           '<PY:VAR2USE>': var2use,
           '<PY:FOLDER2USE>': folder2use,           
           '<PY:START_YEAR>':  f'{year_start_TC}',
           '<PY:END_YEAR>':  f'{year_end_TC}',
           })

## A04: filters using masks
if ('A04' in list2run):
    print('>>>>>>>>>>>>>>>>>>>>>>>>A04')

    matlab_run('A04_filterUsingMasks.m', {
           '<PY:WINDOW_SIZE>': window_size,
           '<PY:MIN_NUM_OBS>': min_num_obs,
           '<PY:VAR2USE>': var2use,
           '<PY:FOLDER2USE>': folder2use,           
           })

## A05: divides profile in two groups:
if ('A05' in list2run):
    print('>>>>>>>>>>>>>>>>>>>>>>>>A05')

    ## A05: Hurricane profiles
    matlab_run('A05_splitHurricaneProfiles.m', {
        '<PY:GRID_VAR_FN>': folder2use + '/Outputs/gridArgoProfHurricane_',
        '<PY:MASK_VALUE>': '0',
        '<PY:MASK_NAME>': 'NoHurMask.csv',
        '<PY:WINDOW_SIZE>': window_size,
        '<PY:MIN_NUM_OBS>': min_num_obs,
        '<PY:VAR2USE>': var2use,
        '<PY:FOLDER2USE>': folder2use,  
           })

    ## A05: Non-hurricane profiles
    matlab_run('A05_splitHurricaneProfiles.m', {
        '<PY:GRID_VAR_FN>': folder2use + '/Outputs/gridArgoProfNonHurricane_',
        '<PY:MASK_VALUE>': '1',
        '<PY:MASK_NAME>': 'NoHurMask.csv',
        '<PY:WINDOW_SIZE>': window_size,
        '<PY:MIN_NUM_OBS>': min_num_obs,
        '<PY:VAR2USE>': var2use,
        '<PY:FOLDER2USE>': folder2use,          
        })

## A06: estimate biases due to: temporal mean, seasonal cycle and trend (amplitude)
if ('A06' in list2run):
    print('>>>>>>>>>>>>>>>>>>>>>>>>A06')
    matlab_run('A06_estimateMeanField.m', {
        '<PY:START_YEAR>':  f'{year_start_TC}',
        '<PY:END_YEAR+1>': f'{year_end_TC+1}',
        '<PY:WINDOW_SIZE>': window_size,
        '<PY:MIN_NUM_OBS>': min_num_obs,
        '<PY:VAR2USE>': var2use,
        '<PY:FOLDER2USE>': folder2use,  
           })

## A07: subtract mean for non-hurricanes
if ('A07' in list2run):
    print('>>>>>>>>>>>>>>>>>>>>>>>>A07')
    matlab_run('A07_subtractMean.m', {
        '<PY:GRID_DATA_FN>':        folder2use + '/Outputs/gridArgoProfFiltered_',
        '<PY:RES_DATA_FN>':         folder2use + '/Outputs/gridArgoRes_',
        '<PY:WINDOW_SIZE>': window_size,
        '<PY:MIN_NUM_OBS>': min_num_obs,
        '<PY:VAR2USE>': var2use,
        '<PY:FOLDER2USE>': folder2use,  
       })
    matlab_run('A07_subtractMean.m', {
        '<PY:GRID_DATA_FN>':        folder2use + '/Outputs/gridArgoProfNonHurricane_',
        '<PY:RES_DATA_FN>':         folder2use + '/Outputs/gridArgoResNonHurricane_',
        '<PY:WINDOW_SIZE>': window_size,
        '<PY:MIN_NUM_OBS>': min_num_obs,
        '<PY:VAR2USE>': var2use,
        '<PY:FOLDER2USE>': folder2use,  
       })

## A08: divides data in months
if ('A08' in list2run):
    print('>>>>>>>>>>>>>>>>>>>>>>>>A08')
    matlab_run('A08_divideDataToMonths.m', {
        '<PY:START_YEAR>': f'{year_start_TC}',
        '<PY:END_YEAR>': f'{year_end_TC}',
        '<PY:WINDOW_SIZE>': window_size,
        '<PY:MIN_NUM_OBS>': min_num_obs,
        '<PY:VAR2USE>': var2use,
        '<PY:FOLDER2USE>': folder2use,  
       })

## A09
if ('A09' in list2run):
    print('>>>>>>>>>>>>>>>>>>>>>>>>A09')
    matlab_run('A09_extendedData.m', {
        '<PY:START_YEAR>': f'{year_start_TC}',
        '<PY:END_YEAR>': f'{year_end_TC}',
        '<PY:WINDOW_SIZE>': window_size,
        '<PY:MIN_NUM_OBS>': min_num_obs,
        '<PY:FOLDER2USE>': folder2use,  
    })

## A10
if ('A10' in list2run):
    print('>>>>>>>>>>>>>>>>>>>>>>>>A10')    
    # NA
    matlab_run('A10_filterLocalMLESpaceTime.m', {
        '<PY:START_YEAR>': f'{year_start_TC}',
        '<PY:END_YEAR>': f'{year_end_TC}',
        '<PY:WINDOW_SIZE>': window_size,
        '<PY:MIN_NUM_OBS>': min_num_obs,
        '<PY:CENTER_MONTH>': center_month,
        '<PY:OCEAN_BASIN>': '_NorthAtlantic',
        '<PY:FOLDER2USE>': folder2use,  
    })
    # WP
    matlab_run('A10_filterLocalMLESpaceTime.m', {
        '<PY:START_YEAR>': f'{year_start_TC}',
        '<PY:END_YEAR>': f'{year_end_TC}',
        '<PY:WINDOW_SIZE>': window_size,
        '<PY:MIN_NUM_OBS>': min_num_obs,
        '<PY:CENTER_MONTH>': center_month,
        '<PY:OCEAN_BASIN>': '_WestPacific',
        '<PY:FOLDER2USE>': folder2use,  
    })

## A11
if ('A11' in list2run):
    print('>>>>>>>>>>>>>>>>>>>>>>>>A11')
    fn = folder2use + '/Outputs/localMLESpaceTime_Depth_{d:03d}_{ws}_{mno}_{wsGP}_{cm:02d}_{sy}_{ey}{ob}.mat'
    for (ob, ob_mesh), depth in product(OB, range(1, depth_layers+1)):
        if not os.path.exists(fn.format(
            d=depth*10,
            ws=window_size,
            mno=min_num_obs,
            wsGP=windowSizeGP,
            cm=int(center_month),
            sy=year_start_TC,
            ey=year_end_TC,
            ob=ob)):
            print(ob, 'layer ',depth)
            current_layer = int(grid_lower) + (depth-1)*int(grid_stride)
            matlab_run('A11_localMLESpaceTime.m', {
                '<PY:CURRENT_LAYER>' : f'{current_layer}',
                '<PY:START_YEAR>': f'{year_start_TC}',
                '<PY:END_YEAR>': f'{year_end_TC}',
                '<PY:DEPTH_INDEX>': f'{depth}',
                '<PY:WINDOW_SIZE>': window_size,
                '<PY:MIN_NUM_OBS>': min_num_obs,
                '<PY:CENTER_MONTH>': center_month,
                '<PY:N_PARPOOL>': n_parpool,
                '<PY:OCEAN_BASIN>': ob,
                '<PY:OB_MESHGRID>': ob_mesh,
                '<PY:WINDOW_SIZE_GP>': windowSizeGP,
                '<PY:FOLDER2USE>': folder2use,  
            })
            
## A12
if ('A12' in list2run):
    print('>>>>>>>>>>>>>>>>>>>>>>>>A12')
    fn = folder2use + '/Outputs/localMLESpaceTime_Depth_{d:03d}_{ws}_{mno}_{wsGP}_{cm:02d}_{sy}_{ey}{ob}.mat'
    for (ob, ob_mesh), depth in product(OB, range(1, depth_layers+1)):
        print(ob, 'layer ', depth)
        current_layer = int(grid_lower) + (depth-1)*int(grid_stride)
        matlab_run('A12_fitLocalMLESpaceTime.m', {
            '<PY:CURRENT_LAYER>' : f'{current_layer}',
            '<PY:GRID_VAR_FN>': folder2use + '/Outputs/gridArgoProfFiltered_',
            '<PY:START_YEAR>': f'{year_start_TC}',
            '<PY:END_YEAR>': f'{year_end_TC}',
            '<PY:DEPTH_INDEX>': f'{depth}',
            '<PY:WINDOW_SIZE>': window_size,
            '<PY:MIN_NUM_OBS>': min_num_obs,
            '<PY:CENTER_MONTH>': center_month,
            '<PY:N_PARPOOL>': n_parpool,
            '<PY:OCEAN_BASIN>': ob,
            '<PY:OB_MESHGRID>': ob_mesh,
            '<PY:WINDOW_SIZE_GP>': windowSizeGP,
            '<PY:VAR2USE>': var2use,
            '<PY:FOLDER2USE>': folder2use,  
        })
