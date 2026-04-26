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

Write-Host "=== Building All CUDA Kernels ===" -ForegroundColor Cyan

$kernels = @(
    @{src="cuda_implementation\kernels\rmsnorm_cuda.cu"; obj="cuda_implementation\kernels\rmsnorm_cuda.obj"},
    @{src="cuda_implementation\kernels\mlp_cuda.cu"; obj="cuda_implementation\kernels\mlp_cuda.obj"},
    @{src="cuda_implementation\kernels\full_attention_cuda.cu"; obj="cuda_implementation\kernels\full_attention_cuda.obj"},
    @{src="cuda_implementation\kernels\lm_head_cuda.cu"; obj="cuda_implementation\kernels\lm_head_cuda.obj"},
    @{src="cuda_implementation\kernels\linear_attention_cuda.cu"; obj="cuda_implementation\kernels\linear_attention_cuda.obj"}
)

$compile_flags = "-x cu -c -allow-unsupported-compiler -use_fast_math -O3 -I`"src\core`" -I`"cuda_implementation\include`""

$nvcc = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe"

foreach ($k in $kernels) {
    Write-Host "Compiling: $($k.src)..." -NoNewline
    $cmd = "$nvcc $compile_flags -o `"$($k.obj)`" `"$($k.src)`""
    Invoke-Expression $cmd 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " FAILED" -ForegroundColor Red
    }
}

Write-Host "`n=== Building CUDA Benchmark Program ===" -ForegroundColor Cyan

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

$all_obj = ($cpu_src | ForEach-Object { $_ -replace '\.cpp$', '.obj' }) + $cuda_obj

Write-Host "Compiling CPU sources..." -ForegroundColor Yellow
foreach ($src in $cpu_src) {
    $obj = $src -replace '\.cpp$', '.obj'
    Write-Host "  cl $src..." -NoNewline
    $cmd = "cl /EHsc /std:c++17 /O2 /I`"src\core`" /I`"cuda_implementation\include`" /c `"$src`" /Fo:`"$obj`""
    Invoke-Expression $cmd 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " FAILED" -ForegroundColor Red
    }
}

Write-Host "`nLinking CUDA benchmark..." -ForegroundColor Yellow
$benchmark_cu = "cuda_implementation\performance_benchmark.cpp"
$output_exe = "build\benchmark_cuda.exe"

$cmd = "cl /EHsc /std:c++17 /O2 /Fe:`"$output_exe`" /link /STACK:268435456 "
foreach ($obj in $all_obj) {
    $cmd += "`"$obj`" "
}
$cmd += "`"$benchmark_cu`" `"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64\cudart.lib`""
Invoke-Expression $cmd 2>$null

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nLinking successful: $output_exe" -ForegroundColor Green
} else {
    Write-Host "`nLinking failed" -ForegroundColor Red
}
