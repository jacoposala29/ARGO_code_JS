close all;
clear;

%% Load data

minNumberOfObs = <PY:MIN_NUM_OBS>;%20;
windowSize = <PY:WINDOW_SIZE>;
var2use  = '<PY:VAR2USE>';
folder2use  = '<PY:FOLDER2USE>';

% Load mean field coefficients (learned from non-hurricane observations)
load([folder2use '/Outputs/meanField_',num2str(windowSize),'_',...
    num2str(minNumberOfObs),'_',var2use,'.mat']);

% Load temperature data: now this is generalized so that we can switch
% between temp and salinity
% inprefix='./Data/gridVarProfHurricane_'
inprefix = '<PY:GRID_DATA_FN>';
load([inprefix,num2str(windowSize),'_',num2str(minNumberOfObs),'_', var2use,'.mat']);

gridDataProf = gridVarObsProf;

%% Subtract mean

profLatAggrSelRounded = roundHalf(profLatAggrSel);
profLongAggrSelRounded = roundHalf(profLongAggrSel);

nProf = length(profLatAggrSelRounded);
gridDataRes = zeros(size(gridDataProf));

nPresGrid = length(presGrid);

progInterval = 20000;

%parfor_progress(ceil(nProf/progInterval));

for iProf = 1:nProf
    profLat = profLatAggrSelRounded(iProf);
    profLong = profLongAggrSelRounded(iProf);
    
    iLat = find(latGrid(1,:) == profLat);
    iLong = find(longGrid(:,1) == profLong);
    
    for iPresGrid = 1:nPresGrid
        betaData = betaGrid(iLong,iLat,:,iPresGrid);
        
        yearDayRatio = fromJulDayToYearDay(profJulDayAggrSel(iProf))/yearLength(profJulDayAggrSel(iProf));

        DataProfHat = betaData(1) ...
                            + betaData(2) * sin(2*pi*1*yearDayRatio) + betaData(3) * cos(2*pi*1*yearDayRatio) ...
                            + betaData(4) * sin(2*pi*2*yearDayRatio) + betaData(5) * cos(2*pi*2*yearDayRatio) ...
                            + betaData(6) * sin(2*pi*3*yearDayRatio) + betaData(7) * cos(2*pi*3*yearDayRatio) ...
                            + betaData(8) * sin(2*pi*4*yearDayRatio) + betaData(9) * cos(2*pi*4*yearDayRatio) ...
                            + betaData(10) * sin(2*pi*5*yearDayRatio) + betaData(11) * cos(2*pi*5*yearDayRatio) ...
                            + betaData(12) * sin(2*pi*6*yearDayRatio) + betaData(13) * cos(2*pi*6*yearDayRatio) ...
                            + betaData(19) * (profJulDayAggrSel(iProf) - midJulDay) ...
                            + betaData(20) * (profJulDayAggrSel(iProf) - midJulDay)^2;
        
        gridDataRes(iProf, iPresGrid) = gridDataProf(iProf, iPresGrid) - DataProfHat;
        
    end
    %     if mod(iProf,progInterval) == 0
    %         parfor_progress;
    %     end
    %
    if mod(iProf,progInterval)==0;disp([num2str(round(iProf/nProf*100)) '% done']);end
end

%parfor_progress(0);

% outprefix = './Data/gridVarResHurricane_';
outprefix = '<PY:RES_DATA_FN>';

gridVarObsRes = gridDataRes;
save([outprefix,num2str(windowSize),'_',num2str(minNumberOfObs),'_',var2use,'.mat'],...
            'profLatAggrSel','profLongAggrSel','profYearAggrSel',...
            'profJulDayAggrSel','profFloatIDAggrSel','profCycleNumberAggrSel',...
            'gridVarObsRes','intStart','intEnd','presGrid', '-v7.3');

exit;
