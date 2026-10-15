param(
    [string]$QdrantUrl = "http://127.0.0.1:16333",
    [int]$ReadyTimeoutSec = 60
)

$ErrorActionPreference = "Stop"

# RAG3_WSL_KEEPALIVE_BEGIN
function Start-Rag3WslKeepalive {
    $keepaliveDir = Join-Path $PSScriptRoot "..\.qdrant_runtime"
    New-Item -ItemType Directory -Force -Path $keepaliveDir | Out-Null
    $pidFile = Join-Path $keepaliveDir "wsl-keepalive.pid"

    if (Test-Path $pidFile) {
        $rawPid = (Get-Content $pidFile -Raw).Trim()
        if ($rawPid -match '^\d+$') {
            $existing = Get-Process -Id ([int]$rawPid) -ErrorAction SilentlyContinue
            if ($existing -and $existing.ProcessName -eq "wsl") {
                return
            }
        }
    }

    $proc = Start-Process -FilePath "wsl.exe" -ArgumentList @("-u", "root", "--", "sleep", "infinity") -WindowStyle Hidden -PassThru
    Set-Content -Path $pidFile -Value $proc.Id -NoNewline
    Start-Sleep -Milliseconds 500

    if (-not (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue)) {
        throw "Failed to start persistent WSL keepalive process"
    }
}
# RAG3_WSL_KEEPALIVE_END

function Test-QdrantReady {
    param([string]$Url)
    try {
        $r = Invoke-WebRequest -Uri ($Url.TrimEnd("/") + "/collections") `
            -UseBasicParsing -TimeoutSec 3
        return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500)
    } catch {
        return $false
    }
}

if (Test-QdrantReady $QdrantUrl) {
    Start-Rag3WslKeepalive
    Write-Output "RAG3_INFRA_READY qdrant=$QdrantUrl source=existing"
    exit 0
}

$composeWin = "C:\python_program\RAG3\compose.yaml"
if (-not (Test-Path $composeWin)) {
    throw "Qdrant is down and compose.yaml is missing: $composeWin"
}

$composeWsl = "/mnt/c/python_program/RAG3/compose.yaml"

Write-Output "RAG3_INFRA_STARTING qdrant=$QdrantUrl"

# WSL2 root avoids recurring docker.sock permission problems.
$cmd = @"
set -eu
if ! docker info >/dev/null 2>&1; then
  if command -v service >/dev/null 2>&1; then
    service docker start >/dev/null 2>&1 || true
  fi
  if command -v systemctl >/dev/null 2>&1; then
    systemctl start docker >/dev/null 2>&1 || true
  fi
fi
docker info >/dev/null
docker compose -f '$composeWsl' up -d qdrant
"@

& wsl.exe -u root -- sh -lc $cmd
if ($LASTEXITCODE -ne 0) {
    throw "Failed to start Qdrant through WSL2 Docker"
}

$deadline = (Get-Date).AddSeconds($ReadyTimeoutSec)
do {
    Start-Sleep -Milliseconds 750
    if (Test-QdrantReady $QdrantUrl) {
        Start-Rag3WslKeepalive
        Write-Output "RAG3_INFRA_READY qdrant=$QdrantUrl source=wsl-docker"
        exit 0
    }
} while ((Get-Date) -lt $deadline)

throw "Qdrant did not become ready at $QdrantUrl within $ReadyTimeoutSec seconds"
