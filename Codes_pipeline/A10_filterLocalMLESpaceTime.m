close all;
clear;

windowSize = <PY:WINDOW_SIZE>;
minNumberOfObs = <PY:MIN_NUM_OBS>;%20;
region = '<PY:OCEAN_BASIN>';
folder2use  = '<PY:FOLDER2USE>';

% Center on September
month = <PY:CENTER_MONTH>;

startYear = <PY:START_YEAR>;
endYear = <PY:END_YEAR>;

[latGrid,longGrid] = meshgrid(linspace(-89.5,89.5,180),linspace(20.5,379.5,360));


if strcmp(region, '_WestPacific')
    % WestPacific _only_ grid
    % Using a landmask prevents contamination from the China-Japan Sea
    maskJohn = ncread('../Inputs/RG_ArgoClim_BATHYMETRY_MASK.nc',...
        'BATHYMETRY_MASK',[1 1 25],[Inf Inf 1]);
    maskJohn(maskJohn == 0) = 1;
    maskJohn = [NaN*ones(360,25) maskJohn NaN*ones(360,25)];
    maskJohn(isnan(maskJohn)) = 0;
    maskJohnT = maskJohn';
    filterMask = maskJohnT; % Everything is shifted by 20 degrees
    landMask(1:180, 21:360) = maskJohnT(1:180, 1:340);
    landMask(1:180, 1:20) = maskJohnT(1:180, 341:360);
elseif strcmp(region, '_NorthAtlantic')
    % NorthPacific _only_ grid
    % Mask out the Pacific to avoid contaminating Gulf of Mexico
    load('../Inputs/OBM.mat')
    NotPacific = 1 - pacificmask;
    filterMask = zeros(180, 360);
    filterMask(1:180, 1:340) = NotPacific(1:180, 21:360);
    filterMask(1:180, 341:360) = NotPacific(1:180, 1:20);
else
    disp('Region unrecognized');
    exit;
end

for iYear = startYear:endYear
    
    S = load([folder2use '/Outputs/Extended/gridArgoRes_',num2str(windowSize),...
        '_',num2str(minNumberOfObs),'_Month_',num2str(month,'%02d'),...
        '_',num2str(iYear),'_extended.mat']);
    
    % Perform data filtering right here
    profLatAggrRounded = roundHalf(S.profLatAggr3Months);
    profLongAggrRounded = roundHalf(S.profLongAggr3Months);
    nProf = length(profLatAggrRounded);
    keep = zeros(1,nProf);
    for iProf = 1:nProf
        latIdx = find(latGrid(1,:) == profLatAggrRounded(iProf));
        longIdx = find(longGrid(:,1) == profLongAggrRounded(iProf));
        
        if ((numel(latIdx) == 0) || (numel(longIdx) == 0))
            keep(iProf) = 0;
        else
            keep(iProf) = filterMask(latIdx,longIdx);
        end
    end
    
    profLatAggr3Months = S.profLatAggr3Months(keep > 0);
    profLongAggr3Months = S.profLongAggr3Months(keep> 0);
    profJulDayAggr3Months = S.profJulDayAggr3Months(keep > 0);
    gridDataRes3Months = S.gridDataRes3Months(keep > 0, :);
    profFloatIDAggr3Months = S.profFloatIDAggr3Months(keep > 0);
    profCycleNumberAggr3Months = S.profCycleNumberAggr3Months(keep > 0);
    intStart = S.intStart;
    intEnd = S.intEnd;
    presGrid = S.presGrid;
    
    save([folder2use '/Outputs/Extended/gridArgoRes_',num2str(windowSize),...
        '_',num2str(minNumberOfObs),'_Month_',num2str(month,'%02d'),...
        '_',num2str(iYear),'_extended_filtered',region,'.mat'],...
        'gridDataRes3Months','profLatAggr3Months',...
        'profLongAggr3Months','profFloatIDAggr3Months',...
        'profCycleNumberAggr3Months', 'profJulDayAggr3Months',...
        'intStart', 'intEnd', 'presGrid');
end

exit;
