param(
    [string]$Python = 'E:/Apps/Conda/miniconda3/envs/uretersam/python.exe',
    [string]$OutputRoot = 'runs/g0_realtime_stage3_20261105/full'
)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $workspace
if (Test-Path -LiteralPath $OutputRoot) { throw "Output exists: $OutputRoot" }
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$failures = @()
foreach ($video in @('video_1','video_2')) {
    # Reverse within-pair order for video 2 to reduce a simple order bias.
    $arms = if ($video -eq 'video_1') { @('v2-serial','v3-serial','v2-pipeline','v3-pipeline') } else { @('v3-serial','v2-serial','v3-pipeline','v2-pipeline') }
    foreach ($arm in $arms) {
        $label = "${arm}_${video}"
        Write-Output "START $label"
        & $Python -B -m experiments.g0_realtime_stage3_20261105.benchmark --arm $arm --video $video --output (Join-Path $OutputRoot $label) *> (Join-Path $OutputRoot "$label.log")
        if ($LASTEXITCODE -ne 0) { $failures += $label }
        Write-Output "FINISH $label exit=$LASTEXITCODE"
    }
}
foreach ($video in @('video_1','video_2')) {
    foreach ($arm in @('v3-serial','v2-pipeline','v3-pipeline')) {
        & $Python -B -m experiments.g0_realtime_20261105.compare --reference (Join-Path $OutputRoot "v2-serial_$video") --candidate (Join-Path $OutputRoot "${arm}_${video}") --output (Join-Path $OutputRoot "equivalence_${arm}_${video}.json")
        if ($LASTEXITCODE -ne 0) { $failures += "equivalence_${arm}_${video}" }
    }
    foreach ($arm in @('v2-serial','v2-pipeline')) {
        & $Python -B -m experiments.g0_realtime_20261105.compare --reference "runs/g0_realtime_stage2_20261105/full/${arm}_$video" --candidate (Join-Path $OutputRoot "${arm}_${video}") --output (Join-Path $OutputRoot "previous_parity_${arm}_${video}.json")
        if ($LASTEXITCODE -ne 0) { $failures += "previous_parity_${arm}_${video}" }
    }
}
if ($failures.Count) { throw "Failed cases: $($failures -join ', ')" }
