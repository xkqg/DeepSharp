# Measures how much of the library the tests actually reach, and fails below 90% of lines or branches.
#
# Why not `dotnet test`: the test project is an executable. xunit v3 runs its suite in-process from its own
# entry point, and `dotnet test` starts a runner that finds nothing to do -- it exits cleanly having executed
# zero tests. Coverage then reports 0%, which reads like a failing gate but is really a dead signal, and the
# same arrangement in the other direction would read like a passing one. So the executable is run, and the
# collector is wrapped around it.
#
#   ./tools/coverage/run.ps1            collect and print the numbers
#   ./tools/coverage/run.ps1 -Check     collect, print, and fail below the threshold

[CmdletBinding()]
param(
    [switch]$Check,
    [double]$LineThreshold = 90,
    [double]$BranchThreshold = 90
)

$ErrorActionPreference = 'Stop'

$root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$settings = Join-Path $PSScriptRoot 'coverage.runsettings'
$outputDir = Join-Path $root 'out\coverage'
$report = Join-Path $outputDir 'coverage.cobertura.xml'

if (-not (Get-Command dotnet-coverage -ErrorAction SilentlyContinue)) {
    Write-Host '==> Installing dotnet-coverage'
    dotnet tool install -g dotnet-coverage
    if ($LASTEXITCODE -ne 0) { throw 'Could not install dotnet-coverage' }
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$testProject = Join-Path $root 'Tst\DeepSharp\DeepSharp.Tests.csproj'
Write-Host '==> Building'
dotnet build $testProject --configuration Release
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }

$dll = Join-Path $root 'Tst\DeepSharp\bin\Release\net10.0\DeepSharp.Tests.dll'
if (-not (Test-Path $dll)) { throw "The test assembly is not where it was expected: $dll" }

Write-Host '==> Collecting coverage'
& dotnet-coverage collect "dotnet exec $dll" --settings $settings --output $report --output-format cobertura
if ($LASTEXITCODE -ne 0) { throw 'Coverage collection failed' }

[xml]$cobertura = Get-Content $report
$lineRate = [double]$cobertura.coverage.'line-rate' * 100
$branchRate = [double]$cobertura.coverage.'branch-rate' * 100

Write-Host ''
Write-Host ("    Lines    {0:N1}%" -f $lineRate)
Write-Host ("    Branches {0:N1}%" -f $branchRate)
Write-Host ''

if (-not $Check) { return }

$failures = @()
if ($lineRate -lt $LineThreshold) {
    $failures += ("line coverage is {0:N1}% and the gate is {1}%" -f $lineRate, $LineThreshold)
}
if ($branchRate -lt $BranchThreshold) {
    $failures += ("branch coverage is {0:N1}% and the gate is {1}%" -f $branchRate, $BranchThreshold)
}

if ($failures.Count -gt 0) {
    foreach ($failure in $failures) { Write-Host "FAIL: $failure" }
    exit 1
}

Write-Host ("PASS: {0:N1}% of lines and {1:N1}% of branches, both above {2}%." -f $lineRate, $branchRate, $LineThreshold)
