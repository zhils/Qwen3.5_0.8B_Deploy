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

Write-Host "=== Linking with Full Path ===" -ForegroundColor Cyan

$link = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35726\bin\Hostx64\x64\link.exe"

if (!(Test-Path $link)) {
    Get-ChildItem "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter "link.exe" -ErrorAction SilentlyContinue | Select-Object -First 5
}

$out = "build\performance_benchmark.exe"
$stack = 268435456

$objs = @(
    "build\performance_benchmark_cuda.obj",
    "build\language_common.obj",
    "build\language_mlp.obj",
    "build\lm_head.obj"
)

$cudart = "`"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64\cudart.lib`""

$obj_list = $objs -join " "

Write-Host "Objects:"
foreach ($o in $objs) { Write-Host "  $o" }
Write-Host ""

$cmd = "`"$link`" /NOLOGO /OUT:`"$out`" /STACK:$stack $obj_list $cudart /SUBSYSTEM:CONSOLE"
Write-Host "Running: $cmd"

Invoke-Expression $cmd 2>&1 | Tee-Object -Variable result

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nLinking successful!" -ForegroundColor Green
    Write-Host "Output: $out" -ForegroundColor Cyan
} else {
    Write-Host "`nLink failed with exit code $LASTEXITCODE" -ForegroundColor Red
    Write-Host "Output:" -ForegroundColor Red
    $result
}
