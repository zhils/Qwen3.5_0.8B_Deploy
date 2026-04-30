$env:VSCMD_SKIP_SENDTELEMETRY = 1
Push-Location "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build"
cmd /c "vcvars64.bat && set" | ForEach-Object {
    if ($_ -match "=") {
        $parts = $_.Split("=", 2)
        [Environment]::SetEnvironmentVariable($parts[0], $parts[1], "Process")
    }
}
Pop-Location
Set-Location d:\deploy\c++deploy

$SRC_CORE = @(
    "src\core\language_common.cpp",
    "src\core\language_mlp.cpp",
    "src\core\language_linear_attn.cpp",
    "src\core\language_full_attn.cpp",
    "src\core\language_backbone.cpp",
    "src\core\token_embedding.cpp",
    "src\core\lm_head.cpp",
    "src\core\sampler.cpp",
    "src\core\mtp_head.cpp",
    "src\core\multimodal_embedding.cpp"
)

$SRC_VISION = @(
    "src\vision\vision_patch_embedding.cpp",
    "src\vision\vision_transformer.cpp"
)

$ALL_SRC = $SRC_CORE + $SRC_VISION

if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
}

Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "Qwen3.5-0.8B Build System" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Yellow

Write-Host "--- Building CPU Benchmark ---`n" -ForegroundColor Cyan
$cmd = "cl /EHsc /std:c++17 /O2 /I`"src\core`" /I`"src\vision`" "
foreach ($s in $ALL_SRC) { $cmd += "`"$s`" " }
$cmd += "`"cuda_implementation\performance_benchmark.cpp`" /Fe:`"build\benchmark_cpu.exe`" /link /STACK:268435456"

Invoke-Expression $cmd

if ($?) {
    Write-Host "`nRunning benchmark..." -ForegroundColor Green
    .\build\benchmark_cpu.exe
} else {
    Write-Host "Build failed" -ForegroundColor Red
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Done!" -ForegroundColor Green
