# Measures how much of the library the tests actually reach, and fails below 90% of lines or branches.
#
# Why not `dotnet test`: every suite is an executable. xunit v3 runs a suite in-process from its own entry
# point, and `dotnet test` starts a runner that finds nothing to do -- it exits cleanly having executed zero
# tests. Coverage then reports 0%, which reads like a failing gate but is really a dead signal, and the same
# arrangement in the other direction would read like a passing one. So each executable is run, and the
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

# A part left from an earlier run is not evidence of this one.
Get-ChildItem $outputDir -Filter '*.cobertura.xml' | Remove-Item

# Every suite, found by the name every suite has rather than listed: a suite the gate never ran would leave its
# package measured by nothing while the total still said PASS. Each is collected on its own and the results are
# merged, so a class two suites both reach is judged on everything that reached it.
$suites = @(Get-ChildItem (Join-Path $root 'Tst') -Recurse -Filter '*.Tests.csproj' |
    Where-Object { $_.FullName -notmatch '[\\/]obj[\\/]' })
if ($suites.Count -eq 0) { throw 'There is no suite under Tst to measure' }

$parts = @()
foreach ($suite in $suites) {
    $name = [IO.Path]::GetFileNameWithoutExtension($suite.Name)

    Write-Host "==> Building $name"
    dotnet build $suite.FullName --configuration Release
    if ($LASTEXITCODE -ne 0) { throw "Build failed: $name" }

    # Every framework the suite was built for, newest first. The newest is measured; every other one is run, and a
    # failure there fails the gate exactly as one on the newest does. Measuring both and merging them was tried and
    # is wrong: the two builds of one assembly merge as one module and most of its branches lose their counts, so
    # every class read as fully branched. The code is one code on both runtimes, so one measurement covers it.
    $builds = @(Get-ChildItem (Join-Path $suite.DirectoryName 'bin/Release') -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName "$name.dll") } |
        Sort-Object { [version]($_.Name -replace '^net', '') } -Descending)
    if ($builds.Count -eq 0) { throw "The test assembly was built for no framework: $name" }

    $measured = $builds[0]
    Write-Host "==> Collecting coverage: $name on $($measured.Name)"
    $part = Join-Path $outputDir "$name.cobertura.xml"
    & dotnet-coverage collect "dotnet exec $(Join-Path $measured.FullName "$name.dll")" --settings $settings --output $part --output-format cobertura
    if ($LASTEXITCODE -ne 0) { throw "Coverage collection failed: $name on $($measured.Name)" }
    $parts += $part

    foreach ($build in $builds | Select-Object -Skip 1) {
        Write-Host "==> Running $name on $($build.Name)"
        & dotnet exec (Join-Path $build.FullName "$name.dll")
        if ($LASTEXITCODE -ne 0) { throw "The suite failed: $name on $($build.Name)" }
    }
}

Write-Host '==> Merging'
& dotnet-coverage merge @parts --output $report --output-format cobertura
if ($LASTEXITCODE -ne 0) { throw 'Merging the coverage failed' }

[xml]$cobertura = Get-Content $report
$lineRate = [double]$cobertura.coverage.'line-rate' * 100
$branchRate = [double]$cobertura.coverage.'branch-rate' * 100

# Per class, not only over everything. A big denominator hides a small class: a hundred well-tested classes
# carry an untested one to a passing total, and the gap only becomes visible the day somebody edits it. The
# compiler's own types -- a lambda's closure, an iterator's state machine -- are pooled into the class they
# were generated for, because that is the class a person wrote and the granularity the rule is about.
$pool = @{}

foreach ($class in $cobertura.SelectNodes('//class')) {
    $owner = $class.name -replace '\.<[^>]*>[a-z]__[A-Za-z0-9_|]*', '' -replace '\.<>c(__DisplayClass[0-9_]*)?', ''

    if (-not $pool.ContainsKey($owner)) {
        $pool[$owner] = [pscustomobject]@{ Lines = 0; LinesHit = 0; Branches = 0; BranchesHit = 0 }
    }

    $counts = $pool[$owner]

    foreach ($line in $class.SelectNodes('lines/line')) {
        $counts.Lines++
        if ([int]$line.hits -gt 0) { $counts.LinesHit++ }

        if ($line.branch -eq 'True' -and $line.'condition-coverage' -match '\((\d+)/(\d+)\)') {
            $counts.BranchesHit += [int]$Matches[1]
            $counts.Branches += [int]$Matches[2]
        }
    }
}

$thin = @()
foreach ($owner in $pool.Keys | Sort-Object) {
    $counts = $pool[$owner]
    if ($counts.Lines -eq 0) { continue }

    $classLines = $counts.LinesHit / $counts.Lines * 100
    $classBranches = 100.0
    if ($counts.Branches -gt 0) { $classBranches = $counts.BranchesHit / $counts.Branches * 100 }

    if ($classLines -lt $LineThreshold -or $classBranches -lt $BranchThreshold) {
        $thin += ("    {0,-56} lines {1,5:N1}%  branches {2,5:N1}%" -f $owner, $classLines, $classBranches)
    }
}

Write-Host ''
Write-Host ("    Lines    {0:N1}%" -f $lineRate)
Write-Host ("    Branches {0:N1}%" -f $branchRate)
Write-Host ("    Classes  {0}, of which {1} below the gate" -f $pool.Count, $thin.Count)
Write-Host ''

if (-not $Check) { return }

$failures = @()
if ($lineRate -lt $LineThreshold) {
    $failures += ("line coverage is {0:N1}% and the gate is {1}%" -f $lineRate, $LineThreshold)
}
if ($branchRate -lt $BranchThreshold) {
    $failures += ("branch coverage is {0:N1}% and the gate is {1}%" -f $branchRate, $BranchThreshold)
}
if ($thin.Count -gt 0) {
    $failures += ("{0} class(es) are below {1}% of lines or {2}% of branches:" -f $thin.Count, $LineThreshold, $BranchThreshold)
    $failures += $thin
}

if ($failures.Count -gt 0) {
    foreach ($failure in $failures) { Write-Host "FAIL: $failure" }
    exit 1
}

Write-Host ("PASS: {0:N1}% of lines and {1:N1}% of branches over everything, and all {2} classes are above {3}% on both." -f $lineRate, $branchRate, $pool.Count, $LineThreshold)
