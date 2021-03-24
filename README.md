# Argo_Salinity

Argo code by Addison Hu, now updated by Jacopo Sala. The original code by Addison is also shared on GitHub. Main changes from that include generalizing the code to run with any variable + plotting some new figures (e.g. to display data availability). Upcoming version will include changes to create/select input data.

This pipeline analyses Argo data to quantify the effect of tropical cyclones (TC) on the upper ocean.

______________________________

The 'Input' folder is designed to contain Argo profile data and TC track data. This Github repository does not include the large .mat files with all the Argo data (e.g. 'argo_data_aggr_2007_2008.mat'). These files can be found in this Google Drive folder: https://drive.google.com/drive/folders/1RehJbNnyKPKy8TXMrTOaTETbx95ORn7r?usp=sharing and need to be added to the 'Input' folder.

The 'Codes_pipeline' folder includes all the scripts needed to run the pipeline. All the user choices (variable to analyze, pressure levels, years, ...) are included in the script 'TC_and_TS_define_param.py'. Right now, all parameters are set to analyze salinity from 10 to 210 dbar (every 5 dbar), using Argo data from 2007 to 2018, and store outputs in the 'Desired_case' folder. This script also includes the path to find Matlab (right now it is set to '/Applications/MATLAB_R2018b.app/bin/matlab'). That will likely need to be changed when the pipeline is run on other platforms. A second README file in the 'Codes_pipeline' folder gives additional details on the single scripts.

The 'Desired_case' folder is where all the outputs from the pipeline will be stored ('Output' subfolder), and is intended to be where the user will have the freedom to add files specific to an analysis case. If the analysis needs to be run for two or more case analysis (e.g. for salinity and temperature), this can easily be done by creating a new folder (e.g. 'Desired_case_2'), with a subfolder called 'Outputs', which needs to have subsubfolders 'Monthly' and 'Extended'. The user will then need to specify the new name of the working directory in 'TC_and_TS_define_param.py' (parameter 'folder2use').

The 'Figure' folder (and its subfolders) is where the pipeline is setup to save its plots.

______________________________

Before running the analysis, it is recommended to remove all the previous files with the commands in 'delete_ALL_previous_files.sh'. Note that the 'Desired_case' folder name might need to be changed based on the name of the working directory chosen by the user.

Next, the whole pipeline can be run using the run_ALL.sh script in the 'Codes_pipeline' folder with the following command:

./run_ALL.sh > log.txt

If only some of the scripts need to be run, this can be specified in the list of scripts to run at the beginning of the run_ALL.sh script. When the analysis is run at a single level, the pipeline should end with script B12, as the following scripts perform an analysis as a function of depth.


