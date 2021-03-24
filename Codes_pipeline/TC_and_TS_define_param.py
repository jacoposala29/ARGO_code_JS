import pandas as pd
import numpy as np
# GENERAL
#data_dir = './Data/'
#year_pairs = (
#        (2007, 2010),
#        (2011, 2014),
#        (2015, 2016),
#        (2017, 2018),
#    )
#var2use       = 'Temperature' 
var2use       = 'Salinity'
case2use = 'noML' # ML if mixed layer case

folder2use = '../SALINITY_5levels'

year_start_TC = 2007
year_end_TC   = 2018
year_pairs = (
        (year_start_TC, 2008),
        (2009,2010),
        (2011,2012),
        (2013,2014),
        (2015,2016),
        (2017,year_end_TC),
    )

grid_lower  = '10'
grid_upper  = '50'
grid_stride = '10'

depth_layers= len(np.arange(int(grid_lower),int(grid_upper)+1,int(grid_stride)))
# depth_layers= 3 #21 This should be computed based on length(grid_lower:grid_stride:grid_upper)


if var2use == 'Salinity':
    clim = 0.4
    unit = 'psu'
elif var2use == 'Temperature': 
    clim = 2.0
    unit = '°C'
elif var2use == 'Density':
    clim = 0.4
    unit = 'TBD'

    
window_size = '8'
windowSizeGP='5'
min_num_obs = '20'
center_month= '9'

matlab_path = '/Applications/MATLAB_R2018b.app/bin/matlab'
n_parpool   = '1' #'8'
# basins
OB = [
    ('_AllBasins',     'meshgrid(linspace(-89.5,89.5,180),linspace(20.5,379.5,360))'),
]

# OB = [
#     ('_NorthAtlantic', 'meshgrid(0.5:70.5,261.5:360.5)'),
#     ('_WestPacific',   'meshgrid(0.5:65.5,105.5:187.5)'),
#     ('_AllBasins',     'meshgrid(linspace(-89.5,89.5,180),linspace(20.5,379.5,360))'),
# ]
# Tracks to include in the analysis
track_dir = '../Inputs'
Basins = [
        ('AL', 'HURDAT_ATLANTIC'),
        ('EP', 'HURDAT_PACIFIC'),
        ('WP', 'JTWC_WESTPACIFIC'),
        ('IO', 'JTWC_INDIANOCEAN'),
        ('SH', 'JTWC_SOUTHERNHEMISPHERE'),
        ]

AL = pd.read_csv(f'{track_dir}/HURDAT_ATLANTIC.csv')
del AL['Unnamed: 0']
EP = pd.read_csv(f'{track_dir}/HURDAT_PACIFIC.csv')
del EP['Unnamed: 0']
WP = pd.read_csv(f'{track_dir}/JTWC_WESTPACIFIC.csv')
del WP['Unnamed: 0']
SH = pd.read_csv(f'{track_dir}/JTWC_SOUTHERNHEMISPHERE.csv')
del SH['Unnamed: 0']
IO = pd.read_csv(f'{track_dir}/JTWC_INDIANOCEAN.csv')
del IO['Unnamed: 0']
Hurricanes_ALL = pd.concat([AL, EP, WP, SH, IO])
