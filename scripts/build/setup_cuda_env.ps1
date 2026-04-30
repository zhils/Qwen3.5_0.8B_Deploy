$env:VSCMD_SKIP_SENDTELEMETRY = 1

Push-Location "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build"
cmd /c "vcvars64.bat && set" | ForEach-Object {
    if ($_ -match "=") {
        $parts = $_.Split("=", 2)
        [Environment]::SetEnvironmentVariable($parts[0], $parts[1], "Process")
    }
}
Pop-Location

$cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
$msvcBin = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64"
$winSys = "C:\Windows\System32"
$winDir = "C:\Windows"

# Keep PATH minimal to avoid nvcc->cl lookup failures
$env:PATH = "$msvcBin;$cudaBin;$winSys;$winDir"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"

