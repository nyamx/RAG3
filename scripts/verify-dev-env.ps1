$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$py = ".venv\Scripts\python.exe"
$wslIpRaw = (wsl.exe -d Ubuntu -u root -- hostname -I).Trim()
$wslIp = ($wslIpRaw -split '\s+')[0]
if (-not $wslIp) { throw "Could not resolve WSL IP." }

$env:RAG3_INFRA_HOST = $wslIp
$env:HF_HOME = (Resolve-Path "data\models\huggingface").Path
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

& $py "scripts\verify-dev-env.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
