param(
    [string]$Python = 'E:/Apps/Conda/miniconda3/envs/uretersam/python.exe',
    [string]$OutputRoot = 'runs/g0_realtime_stage2_20261105/full'
)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $workspace
if (Test-Path -LiteralPath $OutputRoot) { throw "Output exists: $OutputRoot" }
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$failures = @()
foreach ($video in @('video_1','video_2')) {
    foreach ($arm in @('v1-serial','v2-serial','v2-pipeline')) {
        $label = "${arm}_${video}"
        Write-Output "START $label"
        & $Python -B -m experiments.g0_realtime_stage2_20261105.benchmark --arm $arm --video $video --output (Join-Path $OutputRoot $label) *> (Join-Path $OutputRoot "$label.log")
        if ($LASTEXITCODE -ne 0) { $failures += $label }
        Write-Output "FINISH $label exit=$LASTEXITCODE"
    }
}
foreach ($video in @('video_1','video_2')) {
    foreach ($arm in @('v2-serial','v2-pipeline')) {
        & $Python -B -m experiments.g0_realtime_20261105.compare --reference (Join-Path $OutputRoot "v1-serial_$video") --candidate (Join-Path $OutputRoot "${arm}_${video}") --output (Join-Path $OutputRoot "equivalence_${arm}_${video}.json")
        if ($LASTEXITCODE -ne 0) { $failures += "equivalence_${arm}_${video}" }
    }
    & $Python -B -m experiments.g0_realtime_20261105.compare --reference "runs/g0_realtime_20261105/full/fast-runtime_$video" --candidate (Join-Path $OutputRoot "v1-serial_$video") --output (Join-Path $OutputRoot "previous_baseline_parity_${video}.json")
    if ($LASTEXITCODE -ne 0) { $failures += "previous_baseline_parity_${video}" }
}
if ($failures.Count) { throw "Failed cases: $($failures -join ', ')" }
