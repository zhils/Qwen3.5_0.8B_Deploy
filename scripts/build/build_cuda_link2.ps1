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

Write-Host "=== Linking CUDA Benchmark ===" -ForegroundColor Cyan

$link = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35726\bin\Hostx64\x64\link.exe"
$lib = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35726\bin\Hostx64\x64\lib.exe"

$out = "build\benchmark_cuda.exe"
$stack = 268435456

$objs = @(
    "src\core\language_common.obj",
    "src\core\language_mlp.obj",
    "src\core\lm_head.obj",
    "cuda_implementation\kernels\rmsnorm_cuda.obj",
    "cuda_implementation\kernels\mlp_cuda.obj",
    "cuda_implementation\kernels\full_attention_cuda.obj",
    "cuda_implementation\kernels\lm_head_cuda.obj",
    "cuda_implementation\kernels\linear_attention_cuda.obj"
)

$cudart = "`"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64\cudart.lib`""

$obj_str = $objs -join " "

& $link /NOLOGO /OUT:$out /STACK:$stack $obj_str $cudart /SUBSYSTEM:CONSOLE 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nLinking successful!" -ForegroundColor Green
    Write-Host "Output: $out" -ForegroundColor Cyan
} else {
    Write-Host "`nLinking failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
