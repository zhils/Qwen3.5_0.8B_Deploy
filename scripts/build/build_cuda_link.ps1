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

Write-Host "=== Linking CUDA Benchmark ===" -ForegroundColor Cyan

$cpu_src = @(
    "src\core\language_common.cpp",
    "src\core\language_mlp.cpp",
    "src\core\lm_head.cpp"
)

$cuda_obj = @(
    "cuda_implementation\kernels\rmsnorm_cuda.obj",
    "cuda_implementation\kernels\mlp_cuda.obj",
    "cuda_implementation\kernels\full_attention_cuda.obj",
    "cuda_implementation\kernels\lm_head_cuda.obj",
    "cuda_implementation\kernels\linear_attention_cuda.obj"
)

$all_obj = @()
foreach ($src in $cpu_src) {
    $obj = $src -replace '\.cpp$', '.obj'
    $all_obj += $obj
}
$all_obj += $cuda_obj

$benchmark_cu = "cuda_implementation\performance_benchmark.cpp"
$output_exe = "build\benchmark_cuda.exe"

$cmd = "cl /EHsc /std:c++17 /O2 /Fe:`"$output_exe`" /link /STACK:268435456 "
foreach ($obj in $all_obj) {
    $cmd += "`"$obj`" "
}
$cmd += "`"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64\cudart.lib`""

Write-Host "Command: $cmd" -ForegroundColor Yellow
Invoke-Expression $cmd 2>&1 | Tee-Object -Variable output

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n=== Link Error Details ===" -ForegroundColor Red
    $output
}
