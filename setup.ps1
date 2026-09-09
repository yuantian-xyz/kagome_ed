# kagome_ed setup — downloads a PORTABLE Julia into .\julia (nothing is installed
# system-wide; delete this folder to remove everything), then runs a 1-line check.
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$ver = "1.12.6"
$juliaDir = Join-Path $root "julia"
$exe = Join-Path $juliaDir "bin\julia.exe"
if (-not (Test-Path $exe)) {
    $zip = Join-Path $root "julia-$ver-win64.zip"
    $url = "https://julialang-s3.julialang.org/bin/winnt/x64/1.12/julia-$ver-win64.zip"
    Write-Host "Downloading portable Julia $ver (~230 MB) from $url ..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $url -OutFile $zip
    Write-Host "Verifying SHA256 against the official checksum list ..."
    $sums = (Invoke-WebRequest -Uri "https://julialang-s3.julialang.org/bin/checksums/julia-$ver.sha256" -UseBasicParsing).Content
    $line = ($sums -split "`n") | Where-Object { $_ -match "julia-$ver-win64.zip" } | Select-Object -First 1
    $expected = ($line -split "\s+")[0].ToLower()
    $actual = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    if ($expected -ne $actual) { throw "Checksum mismatch for $zip (expected $expected, got $actual)" }
    Write-Host "Checksum OK. Unpacking ..."
    $tmp = Join-Path $root "_julia_tmp"
    if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    $inner = Get-ChildItem $tmp -Directory | Select-Object -First 1
    Move-Item $inner.FullName $juliaDir
    Remove-Item $zip -Force
    Remove-Item $tmp -Recurse -Force
}
$env:JULIA_DEPOT_PATH = Join-Path $root "depot"
New-Item -ItemType Directory -Force -Path (Join-Path $root "output") | Out-Null
Write-Host "Julia check:"
& $exe --startup-file=no -e 'println("  julia ", VERSION, "  threads available: ", Sys.CPU_THREADS, "  RAM: ", round(Sys.total_memory()/2^30, digits=1), " GB")'
Write-Host "Setup OK. Next: run.ps1 smoke / run.ps1 estimate (or double-click batch0.bat)."
