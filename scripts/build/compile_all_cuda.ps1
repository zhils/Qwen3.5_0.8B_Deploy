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

Write-Host "=== Compiling All CUDA Kernels ===" -ForegroundColor Cyan

$kernels = @(
    @{src="cuda_implementation\kernels\rmsnorm_cuda.cu"; name="RMSNorm"},
    @{src="cuda_implementation\kernels\mlp_cuda.cu"; name="MLP"},
    @{src="cuda_implementation\kernels\lm_head_cuda.cu"; name="LMHead"},
    @{src="cuda_implementation\kernels\full_attention_cuda.cu"; name="FullAttention"},
    @{src="cuda_implementation\kernels\linear_attention_cuda.cu"; name="LinearAttention"}
)

$nvcc = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe"

foreach ($k in $kernels) {
    $obj = $k.src -replace '\.cu$', '.obj'
    Write-Host "Compiling $($k.name)..." -NoNewline
    $cmd = "`"$nvcc`" -x cu -c -allow-unsupported-compiler -use_fast_math -O3 -I`"src\core`" -I`"cuda_implementation\include`" -o `"$obj`" `"$($k.src)`""
    Invoke-Expression $cmd 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " FAILED" -ForegroundColor Red
    }
}

Write-Host "`n=== Compiling Benchmark ===" -ForegroundColor Cyan
$cu_file = "cuda_implementation\performance_benchmark.cu"
$output_obj = "build\performance_benchmark_cuda.obj"
Write-Host "Compiling performance_benchmark..." -NoNewline
$cmd = "`"$nvcc`" -x cu -c -allow-unsupported-compiler -use_fast_math -O3 -I`"src\core`" -I`"cuda_implementation\include`" -o `"$output_obj`" `"$cu_file`""
Invoke-Expression $cmd 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host " OK" -ForegroundColor Green
} else {
    Write-Host " FAILED" -ForegroundColor Red
}

Write-Host "`n=== Compiling CPU Sources ===" -ForegroundColor Cyan
$cpu_src = @(
    "src\core\language_common.cpp",
    "src\core\language_mlp.cpp",
    "src\core\lm_head.cpp"
)

foreach ($src in $cpu_src) {
    $obj = [System.IO.Path]::GetFileName($src) -replace '\.cpp$', '.obj'
    Write-Host "Compiling $([System.IO.Path]::GetFileName($src))..." -NoNewline
    $cmd = "cl /EHsc /std:c++17 /O2 /I`"src\core`" /I`"cuda_implementation\include`" /c `"$src`" /Fo:`"build\$obj`""
    Invoke-Expression $cmd 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " FAILED" -ForegroundColor Red
    }
}

Write-Host "`n=== All Compilations Complete ===" -ForegroundColor Cyan
