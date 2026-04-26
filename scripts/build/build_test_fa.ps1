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

Write-Host "=== Building FA-only Test ===" -ForegroundColor Cyan

$nvcc = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe"

& $nvcc -x cu -c -allow-unsupported-compiler -use_fast_math -O3 `
    -I"cuda_implementation\include" `
    -o build\test_fa_only.obj cuda_implementation\test_fa_only.cu

if ($LASTEXITCODE -ne 0) { Write-Host "nvcc test failed!" -ForegroundColor Red; exit 1 }

$linker = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\link.exe"

& $linker /NOLOGO /OUT:"build\test_fa_only.exe" `
    build\test_fa_only.obj `
    cuda_implementation\kernels\full_attention_cuda.obj `
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64\cudart.lib" `
    /SUBSYSTEM:CONSOLE

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nBuild successful! Output: build\test_fa_only.exe" -ForegroundColor Green
} else {
    Write-Host "`nLinking failed!" -ForegroundColor Red
}
