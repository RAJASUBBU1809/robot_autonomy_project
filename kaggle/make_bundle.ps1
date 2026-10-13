# Zip the code + generated Go2 model for upload to Kaggle as a private Dataset.
#
#   powershell -ExecutionPolicy Bypass -File kaggle\make_bundle.ps1
#   powershell -ExecutionPolicy Bypass -File kaggle\make_bundle.ps1 -Init tpms_eng_mjx_v2
#
# -Init adds the newest checkpoint of the given run(s) as init\<run>\ for
# 'train_mjx.py --init-from <run>' (fine-tuning on Kaggle).
#
# Then on kaggle.com: Datasets -> New Dataset -> upload go2_bundle.zip,
# name it "go2-bundle", keep it Private. (After code changes: upload a new version.)
param([string[]]$Init = @())
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

if (-not (Test-Path "models\go2_model\go2.xml")) { throw "Run 'python scripts\build_model.py' first." }

$items = @("pyproject.toml", "README.md", "go2tpms", "scripts", "models")
if (Test-Path "init") { Remove-Item "init" -Recurse -Force }
foreach ($run in $Init) {
    $ckpt = Get-ChildItem "runs\$run\ckpt" -Directory | Where-Object { $_.Name -match '^\d+$' } |
        Sort-Object { [int64]$_.Name } | Select-Object -Last 1
    if (-not $ckpt) { throw "No checkpoint in runs\$run\ckpt" }
    New-Item -ItemType Directory -Force "init\$run\ckpt" | Out-Null
    Copy-Item "runs\$run\config.json" "init\$run\"
    Copy-Item $ckpt.FullName "init\$run\ckpt\" -Recurse
    "Including init\$run\ckpt\$($ckpt.Name)"
}
if ($Init.Count -gt 0) { $items += "init" }
$out = "go2_bundle.zip"
if (Test-Path $out) { Remove-Item $out }
# Windows' built-in tar (bsdtar) writes a standard zip with "/" paths; PowerShell 5.1's
# Compress-Archive writes "\" which breaks the folder structure when unzipped on Linux.
# Full path: Git's GNU tar (if first on PATH) ignores -a and writes a plain tar named .zip.
& "$env:SystemRoot\System32\tar.exe" -a -c -f $out --exclude "__pycache__" @items
if ($LASTEXITCODE -ne 0) { throw "tar failed" }
"Wrote {0} ({1:N1} MB)" -f (Join-Path $root $out), ((Get-Item $out).Length / 1MB)
