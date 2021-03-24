close all;
clear;

minNumberOfObs = <PY:MIN_NUM_OBS>;%20;
windowSize = <PY:WINDOW_SIZE>;
var2use  = '<PY:VAR2USE>';
folder2use  = '<PY:FOLDER2USE>';

load([folder2use '/Outputs/gridArgoResNonHurricane_',num2str(windowSize),'_',...
    num2str(minNumberOfObs),'_',var2use,'.mat']);


nProf = length(profLatAggrSel);

profMonthAggrSel = zeros(1,nProf);
for iProf = 1:nProf
    tempv = datevec(profJulDayAggrSel(iProf));
    profMonthAggrSel(iProf) = tempv(2);
end

gridDataRes = gridVarObsRes;

for iYear = <PY:START_YEAR>:<PY:END_YEAR>
    for iMonth = 1:12
        idx = (profYearAggrSel == iYear & profMonthAggrSel == iMonth);
        
        gridDataResMonth = gridDataRes(idx, :);
        profLatAggrMonth = profLatAggrSel(idx);
        profLongAggrMonth = profLongAggrSel(idx);
        profFloatIDAggrMonth = profFloatIDAggrSel(idx);
        profJulDayAggrMonth = profJulDayAggrSel(idx);
        profCycleNumberAggrMonth = profCycleNumberAggrSel(idx);
        
        save([folder2use '/Outputs/Monthly/gridArgoResNonHurricane_',...
            num2str(windowSize),'_',num2str(minNumberOfObs),...
            '_Month_',num2str(iMonth,'%02d'),'_',num2str(iYear),'.mat'],...
            'gridDataResMonth','profLatAggrMonth','profLongAggrMonth',...
            'profFloatIDAggrMonth','profJulDayAggrMonth',...
            'profCycleNumberAggrMonth','intStart','intEnd','presGrid');
        
        disp(sum(idx));
    end
end
exit;
