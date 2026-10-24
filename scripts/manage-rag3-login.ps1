param(
    [ValidateSet('Inspect','Plan','Prepare','Install','Upgrade')][string]$Operation = 'Inspect'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonw = Join-Path $root '.venv\Scripts\pythonw.exe'
$directPythonw = Join-Path $root '.venv\Scripts\rag3-pythonw-direct.exe'
$python = Join-Path $root '.venv\Scripts\python.exe'
$preparer = Join-Path $root 'scripts\prepare-rag3-direct-pythonw.py'
$entry = Join-Path $root 'scripts\rag3-backend-login.py'
$config = Join-Path $root 'data\backend-login-config.json'
$taskName = 'RAG3 Shared Backend'
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$expectedArguments = ('"' + $entry + '" --run')

function CurrentTask {
    return Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
}
function IsExpectedTask($task) {
    if ($null -eq $task) { return $false }
    $actions = @($task.Actions)
    if ($actions.Count -ne 1) { return $false }
    $actualExe = [System.IO.Path]::GetFullPath($actions[0].Execute)
    $expectedExe = [System.IO.Path]::GetFullPath($directPythonw)
    return [string]::Equals($actualExe, $expectedExe, 'OrdinalIgnoreCase') -and
           [string]::Equals($actions[0].Arguments.Trim(), $expectedArguments, 'Ordinal')
}
function IsLegacyTask($task) {
    if ($null -eq $task) { return $false }
    $actions = @($task.Actions)
    if ($actions.Count -ne 1) { return $false }
    $actualExe = [System.IO.Path]::GetFullPath($actions[0].Execute)
    $legacyExe = [System.IO.Path]::GetFullPath($pythonw)
    return [string]::Equals($actualExe, $legacyExe, 'OrdinalIgnoreCase') -and
           [string]::Equals($actions[0].Arguments.Trim(), $expectedArguments, 'Ordinal')
}
function PrepareDirectPythonw {
    & $python $preparer | Out-Null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $directPythonw)) {
        throw 'Single-process RAG3 venv Pythonw could not be prepared'
    }
}

$existing = CurrentTask
if ($Operation -eq 'Inspect') {
    [pscustomobject]@{
        task_name = $taskName
        registered = ($null -ne $existing)
        matches_expected_action = (IsExpectedTask $existing)
        matches_legacy_action = (IsLegacyTask $existing)
        direct_executable_exists = (Test-Path -LiteralPath $directPythonw)
        config_exists = (Test-Path -LiteralPath $config)
    } | ConvertTo-Json -Compress
    exit 0
}
if (-not (Test-Path -LiteralPath $entry) -or -not (Test-Path -LiteralPath $pythonw) -or
    -not (Test-Path -LiteralPath $python) -or -not (Test-Path -LiteralPath $preparer)) {
    throw 'RAG3 backend login entry or venv executables missing'
}
if ($null -ne $existing -and -not (IsExpectedTask $existing) -and -not (IsLegacyTask $existing)) {
    throw 'Task with the same name has a different action; refusing modification'
}
$port = if ($env:RAG3_MCP_PORT) { [int]$env:RAG3_MCP_PORT } else { 8000 }
$qdrant = if ($env:RAG3_QDRANT_URL) { $env:RAG3_QDRANT_URL } else { 'http://127.0.0.1:16333' }
$write = if ($env:RAG3_MCP_WRITE_ENABLED) { $env:RAG3_MCP_WRITE_ENABLED -eq '1' } else { $true }
$delete = if ($env:RAG3_MCP_DELETE_ENABLED) { $env:RAG3_MCP_DELETE_ENABLED -eq '1' } else { $true }
# Defaults match the legacy enabled stdio plugin. The policy is now backend-wide.
$bootstrapInfra = ($qdrant -eq 'http://127.0.0.1:16333')
$cfg = [ordered]@{
    port = $port; qdrant_url = $qdrant; write_enabled = $write
    delete_enabled = $delete; bootstrap_infra = $bootstrapInfra; tunnel_autostart = $true
}
if (Test-Path -LiteralPath $config) {
    $old = Get-Content -LiteralPath $config -Raw | ConvertFrom-Json
    if (($old | ConvertTo-Json -Compress) -ne (($cfg | ConvertTo-Json -Compress))) {
        throw 'Existing backend login config differs; refusing overwrite'
    }
}
if ($Operation -eq 'Plan') {
    & $python $preparer --check-only | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Base Pythonw/venv version check failed' }
    [pscustomobject]@{
        operation = 'plan'; installed = ($null -ne $existing)
        upgrade_required = (IsLegacyTask $existing)
        direct_executable_exists = (Test-Path -LiteralPath $directPythonw)
        port = $port; qdrant_url = $qdrant; write_enabled = $write
        delete_enabled = $delete; config_exists = (Test-Path -LiteralPath $config)
        task_user = $user
        backend_single_process = (IsExpectedTask $existing)
        after_upgrade_single_process = $true
    } | ConvertTo-Json -Compress
    exit 0
}
if ($Operation -eq 'Prepare') {
    PrepareDirectPythonw
    Write-Output 'DIRECT_PYTHONW_READY scheduled_task_unchanged=true'
    exit 0
}
if ($Operation -eq 'Upgrade') {
    if (-not (IsLegacyTask $existing)) {
        throw 'Upgrade requires the previously installed RAG3 venv redirector task'
    }
    PrepareDirectPythonw
    $action = New-ScheduledTaskAction -Execute $directPythonw -Argument $expectedArguments -WorkingDirectory $root
    Set-ScheduledTask -TaskName $taskName -Action $action | Out-Null
    if (-not (IsExpectedTask (CurrentTask))) { throw 'Scheduled task upgrade could not be verified' }
    Write-Output 'LOGIN_TASK_UPGRADED next_launch=true current_process_unchanged=true'
    exit 0
}
if (IsLegacyTask $existing) {
    throw 'Existing RAG3 task uses the two-process venv redirector; use -Operation Upgrade'
}
PrepareDirectPythonw
if ($null -ne $existing) {
    Write-Output 'ALREADY_REGISTERED unchanged=true'
    exit 0
}
# Install only registers the *future logon* action; does not stop/start current jobs.
# Do not replace a config file or scheduled task owned by anyone else.
if (-not (Test-Path -LiteralPath $config)) {
    $payload = $cfg | ConvertTo-Json -Depth 4
    $tmp = $config + '.stage-d-new'
    if (Test-Path -LiteralPath $tmp) { throw 'Temporary config already exists; refusing overwrite' }
    [System.IO.File]::WriteAllText($tmp, $payload, (New-Object System.Text.UTF8Encoding $false))
    Move-Item -LiteralPath $tmp -Destination $config -ErrorAction Stop
}
& $python $entry --validate --config $config
if ($LASTEXITCODE -ne 0) { throw 'Config validation failed; task was not registered' }
$action = New-ScheduledTaskAction -Execute $directPythonw -Argument $expectedArguments -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) -StartWhenAvailable
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'RAG3 single shared MCP backend (no UI/tray)' | Out-Null
if (-not (IsExpectedTask (CurrentTask))) { throw 'Scheduled task registration could not be verified' }
Write-Output 'LOGIN_TASK_REGISTERED startup_on_next_logon=true no_process_started=true'
