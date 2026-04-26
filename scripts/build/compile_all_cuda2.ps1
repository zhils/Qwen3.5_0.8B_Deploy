. "$PSScriptRoot\setup_cuda_env.ps1"

Set-Location d:\deploy\c++deploy

$kernels = @(
    "cuda_implementation\kernels\rmsnorm_cuda.cu",
    "cuda_implementation\kernels\mlp_cuda.cu",
    "cuda_implementation\kernels\lm_head_cuda.cu",
    "cuda_implementation\kernels\full_attention_cuda.cu",
    "cuda_implementation\kernels\linear_attention_cuda.cu",
    "cuda_implementation\kernels\gpu_sampler_argmax.cu"
)

Write-Host "Compiling CUDA kernels..."
foreach ($k in $kernels) {
    $obj = $k -replace '\.cu$', '.obj'
    Write-Host "  $k -> $obj"
    & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" -x cu -c -allow-unsupported-compiler -use_fast_math -O3 -Isrc\core -Icuda_implementation\include -o $obj $k
}

Write-Host "`nCompiling benchmark..."
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" -x cu -c -allow-unsupported-compiler -use_fast_math -O3 -Isrc\core -Icuda_implementation\include -o build\performance_benchmark_cuda.obj cuda_implementation\performance_benchmark.cu

Write-Host "`nCompiling CPU sources..."
foreach ($src in @("src\core\language_common.cpp", "src\core\language_mlp.cpp", "src\core\lm_head.cpp")) {
    $obj = [System.IO.Path]::GetFileName($src) -replace '\.cpp$', '.obj'
    cl /EHsc /std:c++17 /O2 /Isrc\core /Icuda_implementation\include /c $src /Fobuild\$obj
}

Write-Host "`nDone!"
