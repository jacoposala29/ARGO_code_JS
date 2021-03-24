
function contains() {
    local n=$#
    local value=${!n}
    for ((i=1;i < $#;i++)) {
        if [ "${!i}" == "${value}" ]; then
            echo "y"
            return 0
        fi
    }
    echo "n"
    return 1
}


A=("B00_SlimHurricaneDatabase.py" "B01_MarkHurricaneProfiles.py" "A01" "A02" 
   "A03" "A04" "B02_CreateHurricaneMask.py" "B03_HurricanePairs.py" "A05" "A06" 
   "A07" "A08" "A09" "A10" "A11" "A12" "B04_ProfileDict.py" "B05_AttachVariables.py" 
   "B06_KernelSmoothedEstimates.py" "B08_CreateMleCoefficientDF.py" 
   "B09_DiagonalCovariance.py" "B10_BlockCovariance.py" 
   "B35_TpsLoocvExtended.py" "B34_AnalyzeLoocv.py" 
   "B36_TpsLoocvEstimates.py" "B21_TPS_ThreePanel.py") 

A=("B00_SlimHurricaneDatabase.py" "B01_MarkHurricaneProfiles.py" "A01" "A02" 
   "A03" "A04" "B02_CreateHurricaneMask.py" "B03_HurricanePairs.py" "A05" "A06" 
   "A07" "A08" "A09" "A10" "A11" "A12" "B04_ProfileDict.py" "B05_AttachVariables.py" 
   "B06_KernelSmoothedEstimates.py" "B08_CreateMleCoefficientDF.py" 
   "B09_DiagonalCovariance.py" "B10_BlockCovariance.py" 
   "B35_TpsLoocvExtended.py" "B34_AnalyzeLoocv.py" 
   "B36_TpsLoocvEstimates.py" "B21_TPS_ThreePanel.py") 

if [ $(contains "${A[@]}" "B00_SlimHurricaneDatabase.py") == "y" ]; then
    echo "Running B00_SlimHurricaneDatabase"
    python B00_SlimHurricaneDatabase.py
fi
if [ $(contains "${A[@]}" "B01_MarkHurricaneProfiles.py") == "y" ]; then
    echo "Running B01_MarkHurricaneProfiles"
    python B01_MarkHurricaneProfiles.py
fi
if [ $(contains "${A[@]}" "A01") == "y" ]; then
    echo "Running A01"
    python pipeline_matlab_DG.py A01
fi
if [ $(contains "${A[@]}" "A02") == "y" ]; then
    echo "Running A02"
    python pipeline_matlab_DG.py A02
fi
if [ $(contains "${A[@]}" "A03") == "y" ]; then
    echo "Running A03"
    python pipeline_matlab_DG.py A03
fi
if [ $(contains "${A[@]}" "A04") == "y" ]; then
    echo "Running A04"
    python pipeline_matlab_DG.py A04
fi
if [ $(contains "${A[@]}" "B02_CreateHurricaneMask.py") == "y" ]; then
    echo "Running B02_CreateHurricaneMask"
    python B02_CreateHurricaneMask.py
fi
if [ $(contains "${A[@]}" "B03_HurricanePairs.py") == "y" ]; then
    echo "Running B03_HurricanePairs"
    python B03_HurricanePairs.py
fi
if [ $(contains "${A[@]}" "A05") == "y" ]; then
    echo "Running A05"
    python pipeline_matlab_DG.py A05
fi
if [ $(contains "${A[@]}" "A06") == "y" ]; then
    echo "Running A06"
    python pipeline_matlab_DG.py A06
fi
if [ $(contains "${A[@]}" "A07") == "y" ]; then
    echo "Running A07"
    python pipeline_matlab_DG.py A07
fi
if [ $(contains "${A[@]}" "A08") == "y" ]; then
    echo "Running A08"
    python pipeline_matlab_DG.py A08
fi
if [ $(contains "${A[@]}" "A09") == "y" ]; then
    echo "Running A09"
    python pipeline_matlab_DG.py A09
fi
if [ $(contains "${A[@]}" "A10") == "y" ]; then
    echo "Running A10"
    python pipeline_matlab_DG.py A10
fi
if [ $(contains "${A[@]}" "A11") == "y" ]; then
    echo "Running A11"
    python pipeline_matlab_DG.py A11
fi
if [ $(contains "${A[@]}" "A12") == "y" ]; then
    echo "Running A12"
    python pipeline_matlab_DG.py A12
fi
if [ $(contains "${A[@]}" "B04_ProfileDict.py") == "y" ]; then
    echo "Running B04_ProfileDict"
    python B04_ProfileDict.py
fi
if [ $(contains "${A[@]}" "B05_AttachVariables.py") == "y" ]; then
    echo "Running B05_AttachVariables"
    python B05_AttachVariables.py
fi
if [ $(contains "${A[@]}" "B06_KernelSmoothedEstimates.py") == "y" ]; then
    echo "Running B06_KernelSmoothedEstimates"
    python B06_KernelSmoothedEstimates.py
fi
if [ $(contains "${A[@]}" "B08_CreateMleCoefficientDF.py") == "y" ]; then
    echo "Running B08_CreateMleCoefficientDF"
    python B08_CreateMleCoefficientDF.py
fi
if [ $(contains "${A[@]}" "B09_DiagonalCovariance.py") == "y" ]; then
    echo "Running B09_DiagonalCovariance"
    python B09_DiagonalCovariance.py
fi
if [ $(contains "${A[@]}" "B10_BlockCovariance.py") == "y" ]; then
    echo "Running B10_BlockCovariance"
    python B10_BlockCovariance.py
fi
if [ $(contains "${A[@]}" "B35_TpsLoocvExtended.py") == "y" ]; then
    echo "Running B35_TpsLoocvExtended"
    python B35_TpsLoocvExtended.py
fi
if [ $(contains "${A[@]}" "B34_AnalyzeLoocv.py") == "y" ]; then
    echo "Running B34_AnalyzeLoocv"
    python B34_AnalyzeLoocv.py
fi
if [ $(contains "${A[@]}" "B36_TpsLoocvEstimates.py") == "y" ]; then
    echo "Running B36_TpsLoocvEstimates"
    python B36_TpsLoocvEstimates.py
fi
if [ $(contains "${A[@]}" "B21_TPS_ThreePanel.py") == "y" ]; then
    echo "Running B21_TPS_ThreePanel"
    python B21_TPS_ThreePanel.py
fi
