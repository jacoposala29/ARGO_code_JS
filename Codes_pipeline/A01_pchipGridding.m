close all;
clear;

PATH_TO_GSW='./gsw_matlab_v3_04';
PATH_TO_AGGR_DATA='<PY:DATA_LOC>';
var2use  = '<PY:VAR2USE>';
folder2use  = '<PY:FOLDER2USE>';

addpath(genpath(PATH_TO_GSW));

%% Load data
years = '<PY:YEARS>';
%disp([PATH_TO_AGGR_DATA, 'Argo_data_aggr_', years, '.mat'])
load([PATH_TO_AGGR_DATA, 'Argo_data_aggr_', years, '.mat']);

%% Filter out duplicate profiles
[C,ia,ic] = unique([profLatAggr',profLongAggr',profJulDayAggr'],'rows');

profPresAggrGood = profPresAggr(ia);
profTempAggrGood = profTempAggr(ia);
profPsalAggrGood = profPsalAggr(ia);

profLatAggrGood = profLatAggr(ia);
profLongAggrGood = profLongAggr(ia);
profYearAggrGood = profYearAggr(ia);
profJulDayAggrGood = profJulDayAggr(ia);
profFloatIDAggrGood = profFloatIDAggr(ia);
profCycleNumberAggrGood = profCycleNumberAggr(ia);

%% Starting pressure and end pressure
% Not needed for ML with data at a single level
startPres = cellfun(@min,profPresAggrGood);
endPres = cellfun(@max,profPresAggrGood);

%% Profile selection based on start and end pressure
% Not needed for ML with data at a single level
intStart = <PY:GRID_LOWER>;
intEnd   = <PY:GRID_UPPER>;
selIdx = (startPres >=0 & startPres <= intStart & endPres >= intEnd);

profPresAggrSel = profPresAggrGood(selIdx);
profTempAggrSel = profTempAggrGood(selIdx);
profPsalAggrSel = profPsalAggrGood(selIdx);
profLatAggrSel = profLatAggrGood(selIdx);
profLongAggrSel = profLongAggrGood(selIdx);
profYearAggrSel = profYearAggrGood(selIdx);
profJulDayAggrSel = profJulDayAggrGood(selIdx);
profFloatIDAggrSel = profFloatIDAggrGood(selIdx);
profCycleNumberAggrSel = profCycleNumberAggrGood(selIdx);

%% Compute absolute salinity and conservative and potential temperature, you'll need to have the GSW toolbox in Matlab path to run this section, see http://www.teos-10.org/software.htm

% Convert longitude from 20-380 range to 0-360 range
profLongAggrSelTemp = (profLongAggrSel > 360).*(profLongAggrSel - 360) + (profLongAggrSel <= 360).*profLongAggrSel;

% Calculate absolute salinity -- needed for ML case
profAbsSalAggrSel = cellfun(@gsw_SA_from_SP,profPsalAggrSel, profPresAggrSel,...
            num2cell(profLongAggrSelTemp), num2cell(profLatAggrSel), 'UniformOutput', 0);

% Calculate potential temperature -- needed for ML case
switch var2use
    case 'Temperature'
        profPotTempAggrSel = cellfun(@gsw_pt_from_t,profAbsSalAggrSel,profTempAggrSel,...
            profPresAggrSel,'UniformOutput',0);
end

% disp('Uncomment the part to compute pot temp and abs sal and comment the two lines below here')
% profAbsSalAggrSel  = profPsalAggrSel;
% profPotTempAggrSel = profTempAggrSel;

% Prepare variables for vertical interpolation
presGrid = <PY:GRID_LOWER>:<PY:GRID_STRIDE>:<PY:GRID_UPPER>;
nPresGrid = length(presGrid);
nProf = length(profPresAggrSel);
gridVarObsProf = zeros(nProf, nPresGrid);

% parpool(<PY:N_PARPOOL>)
% parfor_progress(nProf);

% Vertical extrapolation on regular grid
for i = 1:nProf
    press=profPresAggrSel{i};
    switch var2use
        case 'Temperature'
            pottemp=profPotTempAggrSel{i};
            gridVarObsProf(i, :)=pchip(press, pottemp, presGrid);
        case 'Salinity'
            absS   = profAbsSalAggrSel{i};
            gridVarObsProf(i, :)=pchip(press, absS, presGrid);
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
