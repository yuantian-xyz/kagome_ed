# Runs one mode with the portable Julia; everything goes to .\output\<mode>_<timestamp>\
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File run.ps1 <smoke|estimate|batch1> [threads]
param(
    [string]$Mode = "smoke",
    [int]$Threads = 1
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$exe = Join-Path $root "julia\bin\julia.exe"
if (-not (Test-Path $exe)) { Write-Host "Julia not found — run setup.ps1 first."; exit 1 }
$env:JULIA_DEPOT_PATH = Join-Path $root "depot"
$env:OPENBLAS_NUM_THREADS = "1"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$console = Join-Path $root "output\console_$($Mode)_$stamp.txt"
Write-Host "Starting mode '$Mode' with $Threads thread(s) at BelowNormal priority. Console copy: $console"
$args = @("--startup-file=no", "-t", "$Threads", (Join-Path $root "run.jl"), $Mode)
$p = Start-Process -FilePath $exe -ArgumentList $args -WorkingDirectory $root -NoNewWindow -PassThru `
        -RedirectStandardOutput $console -RedirectStandardError (Join-Path $root "output\stderr_$($Mode)_$stamp.txt")
try { $p.PriorityClass = "BelowNormal" } catch {}
$p.WaitForExit()
Get-Content $console | Select-Object -Last 25
Write-Host "Exit code: $($p.ExitCode)"
exit $p.ExitCode
