param([string]$Output = 'runs/g0_realtime_stage5_20261106/native_final')
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $workspace
if (Test-Path -LiteralPath $Output) { throw "Output exists: $Output" }
$vsLocator = 'C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe'
$installation = & $vsLocator -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'Existing MSVC C++ installation not found' }
New-Item -ItemType Directory -Path $Output | Out-Null
$target = (Resolve-Path -LiteralPath $Output).Path
$source = Join-Path $PSScriptRoot 'kernels.cpp'
$environmentScript = Join-Path $installation 'VC/Auxiliary/Build/vcvars64.bat'
$compileCommand = 'call "{0}" >nul && cl.exe /nologo /O2 /fp:strict /EHsc /std:c++17 /LD /GS- "{1}" /Feg0_kernels.dll /Fog0_kernels.obj /link /NODEFAULTLIB /NOENTRY /MACHINE:X64' -f $environmentScript,$source
Push-Location -LiteralPath $target
try {
    & $env:ComSpec /d /c $compileCommand *> 'build.log'
    if ($LASTEXITCODE -ne 0) { Get-Content -LiteralPath 'build.log'; throw 'Native compilation failed' }
    $inspectCommand = 'call "{0}" >nul && dumpbin.exe /imports g0_kernels.dll' -f $environmentScript
    & $env:ComSpec /d /c $inspectCommand *> 'imports.log'
    if ($LASTEXITCODE -ne 0) { throw 'Native import inspection failed' }
    [ordered]@{installation=$installation;flags='/O2 /fp:strict /EHsc /std:c++17 /LD /GS- /link /NODEFAULTLIB /NOENTRY /MACHINE:X64';command=$compileCommand;source=$source;source_sha256=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash;dll_sha256=(Get-FileHash -LiteralPath 'g0_kernels.dll' -Algorithm SHA256).Hash} | ConvertTo-Json | Set-Content -LiteralPath 'build_identity.json' -Encoding utf8
    Get-Content -LiteralPath 'build.log'
} finally { Pop-Location }
