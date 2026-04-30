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

Write-Host "=== Building CUDA Benchmark (Single Command) ===" -ForegroundColor Cyan

$benchmark_cu = "cuda_implementation\performance_benchmark.cpp"
$output_exe = "build\benchmark_cuda.exe"

$cpu_src = @(
    "src\core\language_common.cpp",
    "src\core\language_mlp.cpp",
    "src\core\lm_head.cpp"
)

$src_str = ($cpu_src + $benchmark_cu) -join " "

$cmd = "cl /EHsc /std:c++17 /O2 /Fe:`"$output_exe`" /I`"src\core`" /I`"cuda_implementation\include`" /link /STACK:268435456 `"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64\cudart.lib`" $src_str"

Write-Host "Compiling benchmark..." -ForegroundColor Yellow
Invoke-Expression $cmd 2>&1 | Tee-Object -Variable result

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nBuild successful: $output_exe" -ForegroundColor Green
} else {
    Write-Host "`nBuild failed. Error output:" -ForegroundColor Red
    $result | Select-Object -Last 30
}
