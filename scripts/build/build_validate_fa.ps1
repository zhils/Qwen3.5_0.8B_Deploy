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

$nvcc = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe"

Write-Host "=== Building FA Validation ===" -ForegroundColor Cyan

& $nvcc -x cu -c -allow-unsupported-compiler -use_fast_math -O3 `
    -I"cuda_implementation\include" -I"src\core" `
    -o build\validate_fa_cuda.obj cuda_implementation\validate_fa.cu

if ($LASTEXITCODE -ne 0) { Write-Host "nvcc failed!" -ForegroundColor Red; exit 1 }

$cpu_src = @(
    "src\core\language_common.cpp",
    "src\core\language_full_attn.cpp"
)
foreach ($src in $cpu_src) {
    $obj = [System.IO.Path]::GetFileName($src) -replace '\.cpp$', '_vfa.obj'
    cl /EHsc /std:c++17 /O2 /I"src\core" /c $src /Fo:"build\$obj" 2>$null | Out-Null
}

$linker = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\link.exe"

& $linker /NOLOGO /OUT:"build\validate_fa.exe" /STACK:268435456 `
    "build\validate_fa_cuda.obj" `
    "build\language_common_vfa.obj" "build\language_full_attn_vfa.obj" `
    "build\fa_set_weights_e2e.obj" "build\fa_wrapper_e2e.obj" `
    "cuda_implementation\kernels\full_attention_cuda.obj" `
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64\cudart.lib" `
    /SUBSYSTEM:CONSOLE

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nBuild successful!" -ForegroundColor Green
    .\build\validate_fa.exe
} else {
    Write-Host "`nLinking failed!" -ForegroundColor Red
}
