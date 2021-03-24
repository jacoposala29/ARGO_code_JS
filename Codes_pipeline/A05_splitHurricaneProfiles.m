
close all;
clear;

windowSize = <PY:WINDOW_SIZE>;
minNumberOfObs = <PY:MIN_NUM_OBS>;%20;
var2use  = '<PY:VAR2USE>';
folder2use  = '<PY:FOLDER2USE>';

load([folder2use '/Outputs/gridArgoProfFiltered_',num2str(windowSize), '_',num2str(minNumberOfObs),'_', var2use,'.mat']);
HurMask = dlmread(strcat(folder2use,'/Outputs/<PY:MASK_NAME>'))';%readmatrix

% prefix='./Data/gridVarProfHurricane_'
prefix = '<PY:GRID_VAR_FN>';
mask = (HurMask == <PY:MASK_VALUE>)';

profLatAggrSel      = profLatAggrSel(mask);
profLongAggrSel     = profLongAggrSel(mask);
profYearAggrSel     = profYearAggrSel(mask);
profJulDayAggrSel   = profJulDayAggrSel(mask);
profFloatIDAggrSel  = profFloatIDAggrSel(mask);
profCycleNumberAggrSel = profCycleNumberAggrSel(mask);

gridVarObsProf        = gridVarObsProf(mask, :);
save([prefix,num2str(windowSize), '_', ...
            num2str(minNumberOfObs),'_', var2use,'.mat'],...
            'profLatAggrSel','profLongAggrSel','profYearAggrSel','profJulDayAggrSel',...
            'profFloatIDAggrSel','profCycleNumberAggrSel','gridVarObsProf','intStart',...
            'intEnd','presGrid','-v7.3');
exit;
