. "$PSScriptRoot\setup_cuda_env.ps1"

Set-Location d:\deploy\c++deploy

Write-Host "=== Building E2E Compare Test ===" -ForegroundColor Cyan

Write-Host "Recompiling cuda_engine.cu..." -ForegroundColor Yellow
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" `
    -x cu -c -allow-unsupported-compiler -use_fast_math -O3 `
    -I"cuda_implementation\include" `
    -o build\cuda_engine.obj cuda_implementation\kernels\cuda_engine.cu

Write-Host "Recompiling full_attention_cuda.cu..." -ForegroundColor Yellow
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" `
    -x cu -c -allow-unsupported-compiler -use_fast_math -O3 `
    -I"cuda_implementation\include" `
    -o build\full_attention_cuda.obj cuda_implementation\kernels\full_attention_cuda.cu

& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" `
    -x cu -c -allow-unsupported-compiler -use_fast_math -O3 `
    -I"cuda_implementation\include" -I"src\core" `
    -o build\e2e_compare_cuda.obj cuda_implementation\e2e_compare.cu

if ($LASTEXITCODE -ne 0) { Write-Host "nvcc failed!" -ForegroundColor Red; exit 1 }

Write-Host "Compiling CPU sources..." -ForegroundColor Yellow
$cpu_src = @(
    "src\core\language_common.cpp",
    "src\core\language_mlp.cpp",
    "src\core\language_linear_attn.cpp",
    "src\core\language_full_attn.cpp",
    "src\core\language_backbone.cpp",
    "src\core\lm_head.cpp",
    "cuda_implementation\kernels\fa_wrapper.cpp",
    "cuda_implementation\kernels\fa_set_weights.cpp"
)
foreach ($src in $cpu_src) {
    $obj = [System.IO.Path]::GetFileName($src) -replace '\.cpp$', '_e2e.obj'
    cl /EHsc /std:c++17 /O2 /I"cuda_implementation\include" /I"src\core" /c $src /Fo:"build\$obj" 2>$null | Out-Null
}

$linker = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\link.exe"

& $linker /NOLOGO /OUT:"build\e2e_compare.exe" /STACK:268435456 `
    "build\e2e_compare_cuda.obj" "build\cuda_engine.obj" `
    "build\language_common_e2e.obj" "build\language_mlp_e2e.obj" `
    "build\language_linear_attn_e2e.obj" "build\language_full_attn_e2e.obj" `
    "build\language_backbone_e2e.obj" "build\lm_head_e2e.obj" `
    "build\fa_wrapper_e2e.obj" `
    "build\fa_set_weights_e2e.obj" `
    "cuda_implementation\kernels\rmsnorm_cuda.obj" `
    "cuda_implementation\kernels\mlp_cuda.obj" `
    "cuda_implementation\kernels\lm_head_cuda.obj" `
    "cuda_implementation\kernels\full_attention_cuda.obj" `
    "cuda_implementation\kernels\linear_attention_cuda.obj" `
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64\cudart.lib" `
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64\cublas.lib" `
    /SUBSYSTEM:CONSOLE

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nBuild successful! Output: build\e2e_compare.exe" -ForegroundColor Green
} else {
    Write-Host "`nLinking failed!" -ForegroundColor Red
}
