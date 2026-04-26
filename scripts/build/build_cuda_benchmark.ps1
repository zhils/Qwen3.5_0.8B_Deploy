$env:VSCMD_SKIP_SENDTELEMETRY = 1
Push-Location "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build"
cmd /c "vcvars64.bat && set" | ForEach-Object {
    if ($_ -match "=") {
        $parts = $_.Split("=", 2)
        [Environment]::SetEnvironmentVariable($parts[0], $parts[1], "Process")
    }
}
Pop-Location

$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;$env:PATH"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"

Set-Location d:\deploy\c++deploy

$cu_file = "cuda_implementation\performance_benchmark.cu"
$output = "build\performance_benchmark.exe"

Write-Host "=== Building CUDA Benchmark ===" -ForegroundColor Cyan

& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" `
    -x cu -c `
    -allow-unsupported-compiler `
    -use_fast_math `
    -O3 `
    -I"src\core" `
    -I"cuda_implementation\include" `
    -o "build\performance_benchmark_cuda.obj" `
    $cu_file

if ($LASTEXITCODE -ne 0) {
    Write-Host "CUDA compilation failed" -ForegroundColor Red
    exit 1
}

Write-Host "Compiling CPU sources..." -ForegroundColor Yellow

$cpu_src = @(
    "src\core\language_common.cpp",
    "src\core\language_mlp.cpp",
    "src\core\lm_head.cpp"
)

foreach ($src in $cpu_src) {
    $obj = [System.IO.Path]::GetFileName($src) -replace '\.cpp$', '.obj'
    Write-Host "  cl $src..." -NoNewline
    $cmd = "cl /EHsc /std:c++17 /O2 /I`"src\core`" /I`"cuda_implementation\include`" /c `"$src`" /Fo:`"build\$obj`""
    Invoke-Expression $cmd 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " FAILED" -ForegroundColor Red
    }
}

Write-Host "`nLinking..." -ForegroundColor Yellow

$cmd = "cl /EHsc /Fe:`"$output`" /link /STACK:268435456 "
$cmd += "`"build\performance_benchmark_cuda.obj`" "
$cmd += "`"build\language_common.obj`" "
$cmd += "`"build\language_mlp.obj`" "
$cmd += "`"build\lm_head.obj`" "
$cmd += "`"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64\cudart.lib`""

Invoke-Expression $cmd 2>&1 | Tee-Object -Variable result

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nBuild successful: $output" -ForegroundColor Green
} else {
    Write-Host "`nLink failed" -ForegroundColor Red
    $result
}
