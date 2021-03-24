years = '<PY:YEARS>';
var2use  = '<PY:VAR2USE>';
folder2use  = '<PY:FOLDER2USE>';

prefix = strcat(folder2use,'/Outputs/gridArgoProf');

years_vec = strrep(years,'(','');
years_vec = strrep(years_vec,')','');
years_vec = strrep(years_vec,' ','');

years_vec = strsplit(years_vec,',');
if isempty(years_vec{end})
   years_vec = years_vec(1:end-1);
end
years_start_vec = years_vec(1:2:end);
years_end_vec   = years_vec(2:2:end);

bfr = load([prefix, '_', years_start_vec{1}, '_', years_end_vec{1}, '_', var2use, '.mat']);
if length(years_start_vec)>1
    for i=2:length(years_start_vec)
        clear B
        B = load([prefix, '_', years_start_vec{i}, '_', years_end_vec{i}, '_', var2use, '.mat']);

        gridVarObsProf = cat(1, bfr.gridVarObsProf, B.gridVarObsProf);

        profLatAggrSel = cat(2, bfr.profLatAggrSel, B.profLatAggrSel);
        profLongAggrSel = cat(2, bfr.profLongAggrSel, B.profLongAggrSel);
        profYearAggrSel = cat(2, bfr.profYearAggrSel, B.profYearAggrSel);
        profJulDayAggrSel = cat(2, bfr.profJulDayAggrSel, B.profJulDayAggrSel);
        profFloatIDAggrSel = cat(2, bfr.profFloatIDAggrSel, B.profFloatIDAggrSel);
        profCycleNumberAggrSel = cat(2, bfr.profCycleNumberAggrSel, B.profCycleNumberAggrSel);
        presGrid=bfr.presGrid;
        intStart=bfr.intStart;
        intEnd  =bfr.intEnd;

        save([folder2use '/Outputs/bfr' var2use '.mat'], 'profLatAggrSel','profLongAggrSel','profYearAggrSel',...
                    'profJulDayAggrSel','profFloatIDAggrSel','profCycleNumberAggrSel',...
                    'gridVarObsProf','intStart','intEnd','presGrid', '-v7.3');
        clear bfr
        bfr = load([folder2use '/Outputs/bfr' var2use '.mat']);
    end
else

    gridVarObsProf           = bfr.gridVarObsProf;

    profLatAggrSel         = bfr.profLatAggrSel;
    profLongAggrSel        = bfr.profLongAggrSel;
    profYearAggrSel        = bfr.profYearAggrSel;
    profJulDayAggrSel      = bfr.profJulDayAggrSel;
    profFloatIDAggrSel     = bfr.profFloatIDAggrSel;
    profCycleNumberAggrSel = bfr.profCycleNumberAggrSel;
    presGrid=bfr.presGrid;
    intStart=bfr.intStart;
    intEnd  =bfr.intEnd;

end
save([folder2use '/Outputs/gridArgoProf' var2use '.mat'], 'profLatAggrSel','profLongAggrSel','profYearAggrSel',...
            'profJulDayAggrSel','profFloatIDAggrSel','profCycleNumberAggrSel',...
            'gridVarObsProf','intStart','intEnd','presGrid', '-v7.3');

exit;
