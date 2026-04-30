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

Write-Host "=== Building CPU-only Test (no /GS) ===" -ForegroundColor Cyan

cl /EHsc /std:c++17 /O2 /GS- /I"src\core" `
    cuda_implementation\test_cpu_only.cpp `
    src\core\language_common.cpp src\core\language_mlp.cpp `
    src\core\language_linear_attn.cpp src\core\language_full_attn.cpp `
    src\core\language_backbone.cpp /link /STACK:16777216 /OUT:build\test_cpu_only.exe

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nBuild successful!" -ForegroundColor Green
    .\build\test_cpu_only.exe
} else {
    Write-Host "`nBuild failed!" -ForegroundColor Red
}
