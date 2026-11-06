param(
    [string]$Python = 'E:/Apps/Conda/miniconda3/envs/uretersam/python.exe',
    [string]$OutputRoot = 'runs/g0_realtime_stage10_20261106/full'
)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $workspace
if (Test-Path -LiteralPath $OutputRoot) { throw "Output exists: $OutputRoot" }
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$failures = @()
foreach ($video in @('video_1','video_2')) {
    $arms = if ($video -eq 'video_1') { @('v7-pipeline','v10-pipeline') } else { @('v10-pipeline','v7-pipeline') }
    foreach ($arm in $arms) {
        $label = "${arm}_${video}"
        Write-Output "START $label"
        & $Python -B -m experiments.g0_realtime_stage10_20261106.benchmark --arm $arm --video $video --output (Join-Path $OutputRoot $label) *> (Join-Path $OutputRoot "$label.log")
        if ($LASTEXITCODE -ne 0) { $failures += $label }
        Write-Output "FINISH $label exit=$LASTEXITCODE"
    }
}
if ($failures.Count) { throw "Failed cases before comparison: $($failures -join ', ')" }
foreach ($video in @('video_1','video_2')) {
    & $Python -B -m experiments.g0_realtime_20261105.compare --reference (Join-Path $OutputRoot "v7-pipeline_$video") --candidate (Join-Path $OutputRoot "v10-pipeline_$video") --output (Join-Path $OutputRoot "equivalence_$video.json")
    if ($LASTEXITCODE -ne 0) { $failures += "equivalence_$video" }
    & $Python -B -m experiments.g0_realtime_20261105.compare --reference "runs/g0_realtime_stage8_20261106/full/v7-pipeline_$video" --candidate (Join-Path $OutputRoot "v7-pipeline_$video") --output (Join-Path $OutputRoot "previous_parity_$video.json")
    if ($LASTEXITCODE -ne 0) { $failures += "previous_parity_$video" }
}
if ($failures.Count) { throw "Failed cases: $($failures -join ', ')" }
