close all;
clear;

var2use  = '<PY:VAR2USE>';
folder2use  = '<PY:FOLDER2USE>';
load([folder2use '/Outputs/gridArgoProf' var2use '.mat']);

year_start_TC  = <PY:START_YEAR>;
year_end_TC  = <PY:END_YEAR>;

[latGrid,longGrid] = meshgrid(linspace(-89.5,89.5,180),linspace(20.5,379.5,360));

%% Count number of data points for each month and year within each window

windowSize = <PY:WINDOW_SIZE>;
minNumberOfObs = <PY:MIN_NUM_OBS>;%20;

% Enable wrap around by duplicating boundary data
leftBoundaryIdx = find(profLongAggrSel <= 20 + windowSize);
rightBoundaryIdx = find(profLongAggrSel >= 380 - windowSize);
profLongAggrSel = [profLongAggrSel profLongAggrSel(leftBoundaryIdx) + 360 profLongAggrSel(rightBoundaryIdx) - 360];
profLatAggrSel = [profLatAggrSel profLatAggrSel(leftBoundaryIdx) profLatAggrSel(rightBoundaryIdx)];
profJulDayAggrSel = [profJulDayAggrSel profJulDayAggrSel(leftBoundaryIdx) profJulDayAggrSel(rightBoundaryIdx)];

nGrid = numel(latGrid);

yearMonthCounts = zeros([size(latGrid),numel(year_start_TC:year_end_TC),12]);

%parfor_progress(nGrid);

% Loop across regular grid points
for iGrid = 1:nGrid
    % Select coordinates for this iteration
    latSel = latGrid(iGrid);
    longSel = longGrid(iGrid);
    
    % Bin of lat-lon to consider in this iteration
    latMin = latSel - windowSize;
    latMax = latSel + windowSize;
    longMin = longSel - windowSize;
    longMax = longSel + windowSize;
    
    % Number of profiles in this lat-lon bin
    idx = find(profLatAggrSel > latMin & profLatAggrSel < latMax & profLongAggrSel > longMin & profLongAggrSel < longMax);
    
    % Dates of selected profiles in this lat-lon bin
    profJulDayAggrWindow = profJulDayAggrSel(idx)';    
    
    % Indexes of the lat-lon point being considered
    [iGridSub1,iGridSub2] = ind2sub(size(latGrid),iGrid);
    
    % Store year and month of the profiles selected for this lat-lon bin
    nProf = length(idx);
    profYearAggrWindow = zeros(1,nProf);
    profMonthAggrWindow = zeros(1,nProf);
    for iProf = 1:nProf
        temp = datevec(profJulDayAggrWindow(iProf));
        profYearAggrWindow(iProf) = temp(1);
        profMonthAggrWindow(iProf) = temp(2);
    end
    
    % Create a matrix to store number of selected profiles in each
    % year-month
    years = year_start_TC:year_end_TC; %2007:2010;
    yearMonthCountsWindow = zeros(numel(years),12);
    for iYear=1:numel(years)
        yearMonthCountsWindow(iYear,:) = histcounts(profMonthAggrWindow(profYearAggrWindow == years(iYear)),0.5:12.5);
    end
    
    % Store matrix for this grid point
    yearMonthCounts(iGridSub1,iGridSub2,:,:) = yearMonthCountsWindow;
    %parfor_progress;
    % Print loop progress
    if mod(iGrid,5000)==0;disp([num2str(round(iGrid/nGrid*100)) '% done']);end
end

%parfor_progress(0);
disp([num2str(round(iGrid/nGrid*100)) '% done'])
%% Form data-driven landmask
dataMask = zeros(size(latGrid));

% Loop across all points in regular lat-lon grid
for iGrid = 1:nGrid
    
    % index of current lat-lon point on regular grid
    [iGridSub1,iGridSub2] = ind2sub(size(latGrid),iGrid);
    
    % current lat-lon on regular grid 
    predLat = latGrid(iGrid);
    predLong = longGrid(iGrid);
    
    % Retrieve year-month matrix for the current grid point
    yearMonthCountsWindow = squeeze(yearMonthCounts(iGridSub1,iGridSub2,:,:));
    
    % Create mask: 
    % 0 if there's at least one calendar month with at least minNumberOfObs profiles
    % 1 if ALL calendar months had at least minNumberOfObs profiles
    dataMask(iGrid) = all(sum(yearMonthCountsWindow,1) >= minNumberOfObs);
    
end

% Final mask: discard grid points where not all months had at least
% minNumberOfObs profiles
dataMask(dataMask == 0) = NaN;

% Save mask
save([folder2use '/Outputs/dataMask_',num2str(windowSize), '_', ...
        num2str(minNumberOfObs),'.mat'],...
    'dataMask','latGrid','longGrid','-v7.3');

exit;
