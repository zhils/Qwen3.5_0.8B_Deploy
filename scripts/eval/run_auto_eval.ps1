param(
    [int]$PrefillTokens = 1024,
    [int]$DecodeTokens = 512,
    [int]$Rounds = 3,
    [int]$BatchSize = 1,
    [string]$BuildConfig = "Release",
    [switch]$UseWsl = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Parse-PerformanceOutput {
    param([string]$Text)

    $result = [ordered]@{
        ttft_ms = $null
        prefill_tps = $null
        tpot_ms = $null
        decode_tps = $null
        e2e_tps = $null
        vram_used_mb = $null
    }

    if ($Text -match "TTFT:\s+([0-9.]+)\s+ms") { $result.ttft_ms = [double]$Matches[1] }
    if ($Text -match "Single thrpt:\s+([0-9.]+)\s+tokens/sec") { $result.prefill_tps = [double]$Matches[1] }
    if ($Text -match "TPOT:\s+([0-9.]+)\s+ms/token") { $result.tpot_ms = [double]$Matches[1] }
    if ($Text -match "--- Decode[\s\S]*?Single thrpt:\s+([0-9.]+)\s+tokens/sec") { $result.decode_tps = [double]$Matches[1] }
    if ($Text -match "E2E thrpt:\s+([0-9.]+)\s+tokens/sec") { $result.e2e_tps = [double]$Matches[1] }
    if ($Text -match "GPU VRAM used:\s+([0-9.]+)\s+MB") { $result.vram_used_mb = [double]$Matches[1] }

    return $result
}

function Run-InferenceCommand {
    param(
        [string]$RepoRoot,
        [string]$Command
    )

    if ($UseWsl) {
        $escaped = $Command.Replace('"', '\"')
        return (& wsl bash -lc "cd /mnt/d/deploy/Qwen3.5_0.8B_Deploy && $escaped" | Out-String)
    }

    Push-Location $RepoRoot
    try {
        return (& powershell -NoProfile -Command $Command | Out-String)
    } finally {
        Pop-Location
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$docsDir = Join-Path $repoRoot "docs"
$buildDir = Join-Path $repoRoot "build"

$verifyExe = Join-Path $buildDir "verify_linear_attn_batch"
$v2Exe = Join-Path $buildDir "v2_kernel_accuracy_validate"
$perfExe = Join-Path $buildDir "performance_test"

Push-Location $repoRoot
try {
    if (-not (Test-Path $verifyExe) -or -not (Test-Path $v2Exe) -or -not (Test-Path $perfExe)) {
        & cmake -S . -B build -DCMAKE_BUILD_TYPE=$BuildConfig -DENABLE_CUDA=ON
        if ($LASTEXITCODE -ne 0) { throw "CMake configure failed." }
        & cmake --build build --config $BuildConfig --target verify_linear_attn_batch v2_kernel_accuracy_validate performance_test
        if ($LASTEXITCODE -ne 0) { throw "CMake build failed." }
    }

    $verifyOutput = Run-InferenceCommand -RepoRoot $repoRoot -Command "./build/verify_linear_attn_batch"
    $v2Output = Run-InferenceCommand -RepoRoot $repoRoot -Command "./build/v2_kernel_accuracy_validate"
    $perfOutput = Run-InferenceCommand -RepoRoot $repoRoot -Command "./build/performance_test $PrefillTokens $DecodeTokens $Rounds $BatchSize"

    $verifyPass = ($verifyOutput -match "PASS: Batch output matches serial output")
    $maxDiff = $null
    if ($verifyOutput -match "Max diff:\s+([0-9.eE+-]+)") { $maxDiff = [double]$Matches[1] }
    $avgDiff = $null
    if ($verifyOutput -match "Avg diff:\s+([0-9.eE+-]+)") { $avgDiff = [double]$Matches[1] }

    $v2Pass = ($v2Output -match "ALL TESTS PASSED")
    $perf = Parse-PerformanceOutput -Text $perfOutput

    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $summary = [ordered]@{
        generated_at = $now
        env = [ordered]@{
            prefill_tokens = $PrefillTokens
            decode_tokens = $DecodeTokens
            rounds = $Rounds
            batch_size = $BatchSize
            build_config = $BuildConfig
            use_wsl = [bool]$UseWsl
        }
        accuracy = [ordered]@{
            verify_linear_attn_batch_pass = $verifyPass
            verify_linear_attn_batch_max_diff = $maxDiff
            verify_linear_attn_batch_avg_diff = $avgDiff
            v2_kernel_accuracy_validate_pass = $v2Pass
        }
        performance = $perf
    }

    $pass = $verifyPass -and $v2Pass -and $perf.ttft_ms -ne $null -and $perf.e2e_tps -ne $null

    $summaryJsonPath = Join-Path $docsDir "latest_eval_summary.json"
    $summaryMdPath = Join-Path $docsDir "latest_eval_summary.md"
    $summary.gate_pass = $pass

    ($summary | ConvertTo-Json -Depth 8) | Set-Content -Path $summaryJsonPath -Encoding UTF8

    $md = @(
        "# Latest Auto Eval Summary"
        ""
        "- Generated at: $now"
        "- Gate pass: **$pass**"
        "- Config: prefill=$PrefillTokens, decode=$DecodeTokens, rounds=$Rounds, batch=$BatchSize"
        ""
        "## Accuracy"
        ""
        "| Check | Result | Value |"
        "|---|---|---|"
        "| verify_linear_attn_batch | $verifyPass | max_diff=$maxDiff, avg_diff=$avgDiff |"
        "| v2_kernel_accuracy_validate | $v2Pass | ALL TESTS PASSED required |"
        ""
        "## Performance"
        ""
        "| Metric | Value |"
        "|---|---:|"
        "| TTFT (ms) | $($perf.ttft_ms) |"
        "| Prefill throughput (tok/s) | $($perf.prefill_tps) |"
        "| TPOT (ms/token) | $($perf.tpot_ms) |"
        "| Decode throughput (tok/s) | $($perf.decode_tps) |"
        "| E2E throughput (tok/s) | $($perf.e2e_tps) |"
        "| VRAM used (MB) | $($perf.vram_used_mb) |"
        ""
        "## Raw Logs (truncated)"
        ""
        "### verify_linear_attn_batch"
        '```text'
        ($verifyOutput.Substring(0, [Math]::Min(1200, $verifyOutput.Length)))
        '```'
        ""
        "### v2_kernel_accuracy_validate"
        '```text'
        ($v2Output.Substring(0, [Math]::Min(1200, $v2Output.Length)))
        '```'
    ) -join "`r`n"

    Set-Content -Path $summaryMdPath -Value $md -Encoding UTF8

    Write-Host "Auto eval complete."
    Write-Host "  - $summaryJsonPath"
    Write-Host "  - $summaryMdPath"
    if (-not $pass) { throw "Auto eval gate failed. Check docs/latest_eval_summary.md" }
}
finally {
    Pop-Location
}
