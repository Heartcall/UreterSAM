param(
    [string]$Python = 'E:/Apps/Conda/miniconda3/envs/uretersam/python.exe',
    [string]$OutputRoot = 'runs/g0_realtime_stage4_20261105/full'
)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $workspace
if (Test-Path -LiteralPath $OutputRoot) { throw "Output exists: $OutputRoot" }
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$failures = @()
foreach ($video in @('video_1','video_2')) {
    $arms = if ($video -eq 'video_1') { @('v3-serial','v4-serial') } else { @('v4-serial','v3-serial') }
    foreach ($arm in $arms) {
        $label = "${arm}_${video}"
        Write-Output "START $label"
        & $Python -B -m experiments.g0_realtime_stage4_20261105.benchmark --arm $arm --video $video --output (Join-Path $OutputRoot $label) *> (Join-Path $OutputRoot "$label.log")
        if ($LASTEXITCODE -ne 0) { $failures += $label }
        Write-Output "FINISH $label exit=$LASTEXITCODE"
    }
}
foreach ($video in @('video_1','video_2')) {
    & $Python -B -m experiments.g0_realtime_20261105.compare --reference (Join-Path $OutputRoot "v3-serial_$video") --candidate (Join-Path $OutputRoot "v4-serial_$video") --output (Join-Path $OutputRoot "equivalence_$video.json")
    if ($LASTEXITCODE -ne 0) { $failures += "equivalence_$video" }
    & $Python -B -m experiments.g0_realtime_20261105.compare --reference "runs/g0_realtime_stage3_20261105/full/v3-serial_$video" --candidate (Join-Path $OutputRoot "v3-serial_$video") --output (Join-Path $OutputRoot "previous_parity_$video.json")
    if ($LASTEXITCODE -ne 0) { $failures += "previous_parity_$video" }
}
if ($failures.Count) { throw "Failed cases: $($failures -join ', ')" }
