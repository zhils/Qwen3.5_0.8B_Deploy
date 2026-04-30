param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-CodeFiles([string]$root) {
    $includeExt = @("*.h", "*.hpp", "*.cuh", "*.c", "*.cc", "*.cpp", "*.cu")
    $searchDirs = @(
        (Join-Path $root "src"),
        (Join-Path $root "tests"),
        (Join-Path $root "scripts")
    )

    $files = @()
    foreach ($dir in $searchDirs) {
        if (-not (Test-Path $dir)) { continue }
        $files += Get-ChildItem -Path $dir -Recurse -File -Include $includeExt |
            Where-Object { $_.FullName -notmatch "\\(build|build_.*|third_party|out|external)\\\\" }
    }
    return $files | Sort-Object -Property FullName -Unique
}

if (-not (Get-Command clang-format -ErrorAction SilentlyContinue)) {
    throw "clang-format not found. Please install LLVM/clang-format and retry."
}

$files = Get-CodeFiles -root $RepoRoot
if ($files.Count -eq 0) {
    Write-Host "No source files found to format."
    exit 0
}

Push-Location $RepoRoot
try {
    if ($CheckOnly) {
        $failed = @()
        foreach ($f in $files) {
            & clang-format --dry-run --Werror -style=file "$($f.FullName)"
            if ($LASTEXITCODE -ne 0) {
                $failed += $f.FullName
            }
        }
        if ($failed.Count -gt 0) {
            Write-Host "Formatting check failed for $($failed.Count) file(s):"
            $failed | ForEach-Object { Write-Host "  - $_" }
            exit 1
        }
        Write-Host "Formatting check passed for $($files.Count) file(s)."
        exit 0
    }

    foreach ($f in $files) {
        & clang-format -i -style=file "$($f.FullName)"
        if ($LASTEXITCODE -ne 0) {
            throw "clang-format failed for $($f.FullName)"
        }
    }
    Write-Host "Formatted $($files.Count) file(s) with clang-format."
}
finally {
    Pop-Location
}
