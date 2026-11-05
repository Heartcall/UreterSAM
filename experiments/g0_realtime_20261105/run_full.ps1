param(
    [string]$Python = 'E:/Apps/Conda/miniconda3/envs/uretersam/python.exe',
    [string]$OutputRoot = 'runs/g0_realtime_20261105/full'
)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $workspace
if (Test-Path -LiteralPath $OutputRoot) { throw "Output exists: $OutputRoot" }
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$jobs = @(
    @{Variant='reference'; Video='video_1'},
    @{Variant='fast-runtime'; Video='video_1'},
    @{Variant='reference'; Video='video_2'},
    @{Variant='fast-runtime'; Video='video_2'}
)
$failures = @()
foreach ($job in $jobs) {
    $label = "$($job.Variant)_$($job.Video)"
    $destination = Join-Path $OutputRoot $label
    $logPath = Join-Path $OutputRoot "$label.log"
    Write-Output "START $label"
    & $Python -B -m experiments.g0_realtime_20261105.benchmark --variant $job.Variant --video $job.Video --output $destination *> $logPath
    if ($LASTEXITCODE -ne 0) { $failures += $label }
    Write-Output "FINISH $label exit=$LASTEXITCODE"
}
foreach ($video in @('video_1','video_2')) {
    & $Python -B -m experiments.g0_realtime_20261105.compare --reference (Join-Path $OutputRoot "reference_$video") --candidate (Join-Path $OutputRoot "fast-runtime_$video") --output (Join-Path $OutputRoot "equivalence_$video.json")
    if ($LASTEXITCODE -ne 0) { $failures += "equivalence_$video" }
}
if ($failures.Count) { throw "Failed cases: $($failures -join ', ')" }
