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

Write-Host "=== Compiling fa_wrapper.cpp ===" -ForegroundColor Yellow
cl /EHsc /std:c++17 /O2 /I"cuda_implementation\include" /I"src\core" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include" /c cuda_implementation\kernels\fa_wrapper.cpp /Fo:"build\fa_wrapper_e2e.obj"
Write-Host "cl exit: $LASTEXITCODE"
