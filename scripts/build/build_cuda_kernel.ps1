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

Write-Host "=== CUDA Environment ===" -ForegroundColor Cyan
Write-Host "CUDA_PATH: $env:CUDA_PATH"
Write-Host ""

Write-Host "Testing nvcc..." -ForegroundColor Cyan
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" --version

Write-Host "`n=== Building CUDA RMSNorm Kernel ===" -ForegroundColor Cyan
$cu_file = "cuda_implementation\kernels\rmsnorm_cuda.cu"
$output = "cuda_implementation\kernels\rmsnorm_cuda.obj"

& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" `
    -x cu -c `
    -allow-unsupported-compiler `
    -use_fast_math `
    -O3 `
    -I"src\core" `
    -I"cuda_implementation\include" `
    -o $output `
    $cu_file

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nRMSNorm kernel compiled successfully!" -ForegroundColor Green
    Write-Host "Output: $output"
} else {
    Write-Host "`nCompilation failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
