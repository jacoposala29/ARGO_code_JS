close all;
clear;

cp0 = 3989.244;
rho0 = 1030;

formatIn = 'dd-mmm-yyyy';
startDay = datenum('01-Jan-<PY:START_YEAR>', formatIn);
endDay = datenum('01-Jan-<PY:END_YEAR+1>', formatIn);
minNumberOfObs = <PY:MIN_NUM_OBS>;%20;
windowSize = <PY:WINDOW_SIZE>;

var2use  = '<PY:VAR2USE>';
folder2use  = '<PY:FOLDER2USE>';
disp([var2use ' during:' datestr(startDay) ' to ' datestr(endDay)])

midJulDay = (endDay - startDay) / 2 + startDay;
disp(['midJulDay is ' datestr(midJulDay)])
%% Load integrated profiles

load([folder2use '/Outputs/gridArgoProfNonHurricane_',num2str(windowSize),'_',num2str(minNumberOfObs),'_', var2use,'.mat']);
nPresGrid = length(presGrid);

%% Enable wrap around by duplicating boundary data

leftBoundaryIdx = find(profLongAggrSel <= 20 + windowSize);
rightBoundaryIdx = find(profLongAggrSel >= 380 - windowSize);
profLongAggrSel = [profLongAggrSel profLongAggrSel(leftBoundaryIdx) + 360 profLongAggrSel(rightBoundaryIdx) - 360];
profLatAggrSel = [profLatAggrSel profLatAggrSel(leftBoundaryIdx) profLatAggrSel(rightBoundaryIdx)];
profJulDayAggrSel = [profJulDayAggrSel profJulDayAggrSel(leftBoundaryIdx) profJulDayAggrSel(rightBoundaryIdx)];

gridDataProf = [gridVarObsProf; gridVarObsProf(leftBoundaryIdx, :); gridVarObsProf(rightBoundaryIdx, :)];

%% Calculate mean field using a moving window

[latGrid,longGrid] = meshgrid(linspace(-89.5,89.5,180),linspace(20.5,379.5,360));

nGrid = numel(latGrid);

%% Modify to do all at once
betaGrid = zeros([size(latGrid),20,nPresGrid]);

load([folder2use '/Outputs/dataMask_',num2str(windowSize),'_',num2str(minNumberOfObs),'.mat']);

mask = dataMask;

tic;

%parfor_progress(nGrid);% not showing the progress bar since it slows down
%the script

for iGrid = 1:nGrid
    latSel = latGrid(iGrid);
    longSel = longGrid(iGrid);

    latMin = latSel - windowSize;
    latMax = latSel + windowSize;
    longMin = longSel - windowSize;
    longMax = longSel + windowSize;
    
    idx = find(profLatAggrSel > latMin & profLatAggrSel < latMax & profLongAggrSel > longMin & profLongAggrSel < longMax);
    
    latIdx = find(latGrid(1,:) == latSel);
    longIdx = find(longGrid(:,1) == longSel);    
    goodPixel = ~isnan(mask(longIdx,latIdx));
    
    % Need at least 20 data points to estimate the regression coefficients, also do not compute mean if outside land/datamask
    if length(idx) < 20 || ~goodPixel
        %parfor_progress
        if mod(iGrid,5000)==0;disp([num2str(round(iGrid/nGrid*100)) '% done']);end
        continue; 
    end
    
    profJulDayAggrWindow = profJulDayAggrSel(idx)';
    profLatAggrWindow = profLatAggrSel(idx)';
    profLongAggrWindow = profLongAggrSel(idx)';
    profYearDayAggrWindow = fromJulDayToYearDay(profJulDayAggrWindow);
    profYearLengthAggrWindow = yearLength(profJulDayAggrWindow);

    % Iterate and fit over all depths
    for iPresGrid = 1:nPresGrid
        gridDataProfWindow = gridDataProf(idx, iPresGrid);
        
        %% NoTrend
        XWindow = [ones(length(profJulDayAggrWindow),1) ...
           sin(2*pi*1*profYearDayAggrWindow./profYearLengthAggrWindow) cos(2*pi*1*profYearDayAggrWindow./profYearLengthAggrWindow) ...
           sin(2*pi*2*profYearDayAggrWindow./profYearLengthAggrWindow) cos(2*pi*2*profYearDayAggrWindow./profYearLengthAggrWindow) ...
           sin(2*pi*3*profYearDayAggrWindow./profYearLengthAggrWindow) cos(2*pi*3*profYearDayAggrWindow./profYearLengthAggrWindow) ...
           sin(2*pi*4*profYearDayAggrWindow./profYearLengthAggrWindow) cos(2*pi*4*profYearDayAggrWindow./profYearLengthAggrWindow) ...
           sin(2*pi*5*profYearDayAggrWindow./profYearLengthAggrWindow) cos(2*pi*5*profYearDayAggrWindow./profYearLengthAggrWindow) ...
           sin(2*pi*6*profYearDayAggrWindow./profYearLengthAggrWindow) cos(2*pi*6*profYearDayAggrWindow./profYearLengthAggrWindow) ...
           (profLatAggrWindow-latSel) (profLongAggrWindow-longSel) (profLatAggrWindow-latSel).*(profLongAggrWindow-longSel) ...
           (profLatAggrWindow-latSel).^2 (profLongAggrWindow-longSel).^2];
            
        betaWindow = XWindow\gridDataProfWindow;
        betaWindow = [betaWindow; 0; 0]; % Put zero coefficients in place of the time trend terms
        [iGridSub1,iGridSub2] = ind2sub(size(latGrid),iGrid);
        betaGrid(iGridSub1,iGridSub2,:,iPresGrid) = betaWindow;
    end
    %length(idx)
    %parfor_progress;
    if mod(iGrid,5000)==0;disp([num2str(round(iGrid/nGrid*100)) '% done']);end
end

%parfor_progress(0);
disp([num2str(round(iGrid/nGrid*100)) '% done'])
toc;

save([folder2use '/Outputs/meanField_',num2str(windowSize),'_',num2str(minNumberOfObs),...
    '_',var2use,'.mat'],'betaGrid','latGrid','longGrid','midJulDay','presGrid', '-v7.3');

exit;
