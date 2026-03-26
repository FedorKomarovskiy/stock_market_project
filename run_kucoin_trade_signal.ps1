param(
    [string]$ProjectDir = "",
    [string]$PythonExe = "",
    [string]$Config = "",
    [string]$ModelPath = "",
    [string]$EnvFile = "",
    [ValidateSet("train", "shadow", "live")]
    [string]$Mode = "shadow",
    [switch]$RunRealOrder,
    [switch]$Once,
    [switch]$ForceTrain,
    [switch]$TrainIfMissing
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONIOENCODING = "utf-8"
cmd /c chcp 65001 > $null

$pythonPrefixArgs = @()
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($ProjectDir)) {
    $ProjectDir = $scriptDir
}
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $venvPython = Join-Path $ProjectDir "venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $PythonExe = $venvPython
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $PythonExe = "python"
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $PythonExe = "py"
        $pythonPrefixArgs = @("-3")
    } else {
        $PythonExe = "python"
    }
}
if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = Join-Path $ProjectDir "config\micro_near_v1_1m.json"
}
if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = Join-Path $ProjectDir "models\near_basis_qlearning.json"
}
if ([string]::IsNullOrWhiteSpace($EnvFile)) {
    $EnvFile = Join-Path $ProjectDir ".runtime\kucoin.env"
}

$runnerScript = Join-Path $ProjectDir "run_trade_signal.py"
$logDir = Join-Path $ProjectDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (-not (Test-Path $PythonExe) -and -not (Get-Command $PythonExe -ErrorAction SilentlyContinue)) {
    throw "Python command not found: $PythonExe"
}
if (-not (Test-Path $runnerScript)) {
    throw "Runner script not found: $runnerScript"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "kucoin_trade_signal_$timestamp.log"

$args = @(
    $runnerScript,
    "--config", $Config,
    "--model-path", $ModelPath,
    "--env-file", $EnvFile,
    "--mode", $Mode
)

if ($RunRealOrder) {
    $args += "--run-real-order"
}
if ($Once) {
    $args += "--once"
}
if ($ForceTrain) {
    $args += "--force-train"
}
if ($TrainIfMissing) {
    $args += "--train-if-missing"
}

Write-Host "Python       :" $PythonExe
Write-Host "Runner script:" $runnerScript
Write-Host "Config       :" $Config
Write-Host "ModelPath    :" $ModelPath
Write-Host "EnvFile      :" $EnvFile
Write-Host "Mode         :" $Mode
Write-Host "RunRealOrder :" $RunRealOrder.IsPresent
Write-Host "Once         :" $Once.IsPresent
Write-Host "ForceTrain   :" $ForceTrain.IsPresent
Write-Host "TrainIfMissing:" $TrainIfMissing.IsPresent
Write-Host "Log file     :" $logPath

& $PythonExe @pythonPrefixArgs @args 2>&1 | Tee-Object -FilePath $logPath
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    throw "run_trade_signal.py finished with exit code $exitCode"
}

Write-Host "Done. ExitCode=0"
