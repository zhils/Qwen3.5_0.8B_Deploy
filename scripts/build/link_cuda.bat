@echo off
set VSCMD_SKIP_SENDTELEMETRY=1
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"

cd /d d:\deploy\c++deploy

set OUT=build\performance_benchmark.exe
set STACK=268435456

set OBJS=build\performance_benchmark_cuda.obj build\language_common.obj build\language_mlp.obj build\lm_head.obj cuda_implementation\kernels\rmsnorm_cuda.obj cuda_implementation\kernels\mlp_cuda.obj cuda_implementation\kernels\lm_head_cuda.obj cuda_implementation\kernels\full_attention_cuda.obj cuda_implementation\kernels\linear_attention_cuda.obj
set LIBS="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64\cudart.lib"

link /NOLOGO /OUT:%OUT% /STACK:%STACK% %OBJS% %LIBS% /SUBSYSTEM:CONSOLE

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Linking successful!
    echo Output: %OUT%
    echo ========================================
) else (
    echo.
    echo Linking failed with exit code %ERRORLEVEL%
)
