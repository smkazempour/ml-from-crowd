param(
    [string]$Bundle = (Split-Path -Parent $PSScriptRoot),
    [string]$Python = 'python',
    [string]$Output = 'runs',
    [ValidateRange(1, 128)][int]$Workers = 24,
    [ValidateRange(1, 128)][int]$Threads = 2,
    [double]$MemoryPerWorkerGiB = 6,
    [double]$ReserveGiB = 24,
    [ValidateSet('nn3', 'depth', 'width')][string[]]$Phases = @('nn3', 'depth'),
    [switch]$Fresh
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Bundle = (Resolve-Path -LiteralPath $Bundle).Path
Set-Location -LiteralPath $Bundle
$env:OMP_NUM_THREADS = "$Threads"
$env:MKL_NUM_THREADS = "$Threads"
$env:OPENBLAS_NUM_THREADS = "$Threads"
$env:NUMEXPR_NUM_THREADS = "$Threads"
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'

function Invoke-StudyPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE : $($Arguments -join ' ')"
    }
}

$checkArguments = @('-B', '-m', 'server_nn.check_environment', '--workers', "$Workers", '--threads', "$Threads",
                    '--memory-per-worker-gib', "$MemoryPerWorkerGiB", '--reserve-gib', "$ReserveGiB")
$trainExtra = @()
if ($Fresh) {
    $checkArguments += '--allow-runtime-difference'
    $trainExtra += '--no-reuse-local'
}
Invoke-StudyPython -Arguments $checkArguments
Invoke-StudyPython -Arguments @('-B', '-m', 'server_nn.verify', '--bundle', '.')
if (-not $Fresh) {
    Invoke-StudyPython -Arguments @('-B', '-m', 'server_nn.verify', '--bundle', '.', '--bridge', '--threads', "$Threads")
}
foreach ($phase in $Phases) {
    $trainArguments = @('-B', '-u', '-m', 'server_nn.train', '--bundle', '.', '--output', $Output,
                        '--phase', $phase, '--workers', "$Workers", '--threads', "$Threads",
                        '--memory-per-worker-gib', "$MemoryPerWorkerGiB", '--reserve-gib', "$ReserveGiB") + $trainExtra
    Invoke-StudyPython -Arguments $trainArguments
    Invoke-StudyPython -Arguments @('-B', '-u', '-m', 'server_nn.evaluate', '--bundle', '.', '--output', $Output, '--phase', $phase)
}
