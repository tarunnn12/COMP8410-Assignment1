# Run the quick (smoke test) version of one pair's evaluation notebook into a separate folder, for a first check or a demonstration.
# Usage (PowerShell, from the repository root):   .\tools\run_quick.ps1 -Pair p7
#   Pair is one of: p2 p4 p6 p7. Results go to demo_runs\<pair>\results (the full results are not touched).
# The kernel and environment variables are the ones listed in each pair README.
param([Parameter(Mandatory = $true)][ValidateSet("p2", "p4", "p6", "p7")][string]$Pair)

$root = Split-Path -Parent $PSScriptRoot
$map = @{
    p2 = @{ dir = "p2_3d_reconstruction";  nb = "p2_3d_reconstruction_eval.ipynb";  var = "P2"; kernel = "comp8410-gen" }
    p4 = @{ dir = "p4_agents";             nb = "p4_agents_eval.ipynb";             var = "P4"; kernel = "comp8410-p4_agents" }
    p6 = @{ dir = "p6_lmm_vqa";            nb = "p6_lmm_vqa_eval.ipynb";            var = "P6"; kernel = "comp8410-gen" }
    p7 = @{ dir = "p7_safety";             nb = "p7_safety_eval.ipynb";             var = "P7"; kernel = "comp8410-gen" }
}
$m = $map[$Pair]
$out = Join-Path $root "demo_runs\$Pair"; New-Item -ItemType Directory -Force $out | Out-Null
Set-Item "Env:$($m.var)_QUICK" "1"
Set-Item "Env:$($m.var)_RESULTS" (Join-Path $out "results")
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Join-Path $root "pairs\$($m.dir)\src")
$t = Get-Date
python -m nbconvert --to notebook --execute $m.nb --ExecutePreprocessor.kernel_name=$($m.kernel) --ExecutePreprocessor.timeout=-1 --output-dir $out --output "$Pair`_quick_executed.ipynb"
"Finished in {0:N1} minutes. Executed notebook and results: {1}" -f ((Get-Date) - $t).TotalMinutes, $out
