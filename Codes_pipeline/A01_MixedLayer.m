close all;
clear;

PATH_TO_GSW='./gsw_matlab_v3_04';
PATH_TO_AGGR_DATA='<PY:DATA_LOC>';
var2use  = '<PY:VAR2USE>';
folder2use  = '<PY:FOLDER2USE>';

addpath(genpath(PATH_TO_GSW));

%% Load data
years = '<PY:YEARS>';
load([PATH_TO_AGGR_DATA, 'Argo_data_aggr_', years, '.mat']);

%% Filter out duplicate profiles
[C,ia,ic] = unique([profLatAggr',profLongAggr',profJulDayAggr'],'rows');

profLatAggrSel = profLatAggr(ia);
profLongAggrSel = profLongAggr(ia);
profYearAggrSel = profYearAggr(ia);
profJulDayAggrSel = profJulDayAggr(ia);
profFloatIDAggrSel = profFloatIDAggr(ia);
profCycleNumberAggrSel = profCycleNumberAggr(ia);

presGrid = <PY:GRID_LOWER>:<PY:GRID_STRIDE>:<PY:GRID_UPPER>;
nPresGrid = length(presGrid);
nProf = length(profLatAggrSel);
gridVarObsProf = zeros(nProf, nPresGrid);

intStart = <PY:GRID_LOWER>;
intEnd   = <PY:GRID_UPPER>;

% parpool(<PY:N_PARPOOL>)
% parfor_progress(nProf);

% Vertical extrapolation on regular grid
for i = 1:nProf
    switch var2use
        case 'Temperature'
            profTempAggrSel = profTempAggr(ia);
            gridVarObsProf(i, :) = profTempAggrSel{i};
        case 'Salinity'
            profPsalAggrSel = profPsalAggr(ia);
            gridVarObsProf(i, :)= profPsalAggrSel{i};
        case 'Depth'
            profDeptAggrSel = profDeptAggr(ia);
            gridVarObsProf(i, :) = profDeptAggrSel{i};
        case 'Density'
            profPdenAggrSel = profPdenAggr(ia);
            gridVarObsProf(i, :) = profPdenAggrSel{i};
    end
    % parfor_progress;
end
% parfor_progress(0);

% Save outputs
save([folder2use,'/Outputs/gridArgoProf_',years,'_',var2use,'.mat'],'profLatAggrSel','profLongAggrSel',...
            'profYearAggrSel','profJulDayAggrSel','profFloatIDAggrSel',...
            'profCycleNumberAggrSel','gridVarObsProf',...
            'intStart','intEnd','presGrid','-v7.3');

exit;
