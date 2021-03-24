
close all;
clear;

var2use  = '<PY:VAR2USE>';
folder2use  = '<PY:FOLDER2USE>';
load([folder2use '/Outputs/gridArgoProf' var2use '.mat']);

windowSize = <PY:WINDOW_SIZE>;
minNumberOfObs = <PY:MIN_NUM_OBS>;%20;

load([folder2use '/Outputs/dataMask_',num2str(windowSize), '_',num2str(minNumberOfObs),'.mat']);

mask = dataMask;
profLatAggrRounded = roundHalf(profLatAggrSel);
profLongAggrRounded = roundHalf(profLongAggrSel);

nProf = length(profLatAggrRounded);
keep = zeros(1,nProf);

for iProf = 1:nProf
    latIdx = find(latGrid(1,:) == profLatAggrRounded(iProf));
    longIdx = find(longGrid(:,1) == profLongAggrRounded(iProf));
    
    keep(iProf) = ~isnan(mask(longIdx,latIdx));
end

disp(sum(keep));
disp(nProf);
disp(sum(keep)/nProf);

keep = logical(keep);

profLatAggrSel = profLatAggrSel(keep);
profLongAggrSel = profLongAggrSel(keep);
profYearAggrSel = profYearAggrSel(keep);
profJulDayAggrSel = profJulDayAggrSel(keep);
profFloatIDAggrSel = profFloatIDAggrSel(keep);
profCycleNumberAggrSel = profCycleNumberAggrSel(keep);

gridVarObsProf = gridVarObsProf(keep, :);
save([folder2use '/Outputs/gridArgoProfFiltered_',num2str(windowSize), ...
            '_', num2str(minNumberOfObs),'_', var2use, '.mat'],...
            'profLatAggrSel','profLongAggrSel','profYearAggrSel',...
            'profJulDayAggrSel','profFloatIDAggrSel','profCycleNumberAggrSel',...
            'gridVarObsProf','intStart','intEnd','presGrid','-v7.3');

exit;
