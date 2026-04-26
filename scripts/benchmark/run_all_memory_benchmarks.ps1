param(
    [int]$KvDecodeSteps = 100,
    [int]$BatchSteps = 50,
    [int]$PagedMaxSeqLen = 2048,
    [int]$PagedPageSize = 64,
    [int]$PagedMaxPages = 128,
    [string]$BuildConfig = "Release"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$buildDir = Join-Path $repoRoot "build"
$docsDir = Join-Path $repoRoot "docs"
$summaryMd = Join-Path $docsDir "latest_benchmark_summary.md"
$summaryCsv = Join-Path $docsDir "latest_benchmark_summary.csv"

$kvExe = Join-Path $buildDir "$BuildConfig\kv_int8_benchmark.exe"
$batchExe = Join-Path $buildDir "$BuildConfig\batch_poc_benchmark.exe"
$pagedExe = Join-Path $buildDir "$BuildConfig\paged_kv_benchmark.exe"

function Ensure-BenchmarkBuild {
    if ((Test-Path $kvExe) -and (Test-Path $batchExe) -and (Test-Path $pagedExe)) {
        return
    }
    Push-Location $repoRoot
    try {
        & cmake -B build -DCMAKE_BUILD_TYPE=$BuildConfig -DENABLE_CUDA=ON
        if ($LASTEXITCODE -ne 0) { throw "CMake configure failed." }
        & cmake --build build --config $BuildConfig --target kv_int8_benchmark batch_poc_benchmark paged_kv_benchmark
        if ($LASTEXITCODE -ne 0) { throw "CMake build failed." }
    } finally {
        Pop-Location
    }
}

function Get-ValueFromKvCsv([array]$rows, [string]$metric, [string]$col) {
    $row = $rows | Where-Object { $_.metric -eq $metric } | Select-Object -First 1
    if (-not $row) { throw "Metric $metric not found in kv csv." }
    return [double]$row.$col
}

function Parse-BatchOutput([string]$text) {
    $forward = @{}
    $sequential = @{}
    $gain = @{}

    $mode = ""
    foreach ($line in ($text -split "`r?`n")) {
        if ($line -match "^\[forward_batch Mode\]") { $mode = "forward"; continue }
        if ($line -match "^\[sequential Baseline Mode\]") { $mode = "sequential"; continue }
        if ($line -match "^\[API Gain: forward_batch / sequential\]") { $mode = "gain"; continue }

        if ($mode -eq "forward" -and $line -match "^\s*(\d+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)x") {
            $batch = [int]$Matches[1]
            $forward[$batch] = @{
                avg_step_ms = [double]$Matches[2]
                throughput_tps = [double]$Matches[3]
                speedup = [double]$Matches[4]
            }
        } elseif ($mode -eq "sequential" -and $line -match "^\s*(\d+)\s+\|\s+([0-9.]+)\s*$") {
            $batch = [int]$Matches[1]
            $sequential[$batch] = [double]$Matches[2]
        } elseif ($mode -eq "gain" -and $line -match "^\s*(\d+)\s+\|\s+([0-9.]+)x") {
            $batch = [int]$Matches[1]
            $gain[$batch] = [double]$Matches[2]
        }
    }

    return @{
        forward = $forward
        sequential = $sequential
        gain = $gain
    }
}

function Parse-PagedOutput([string]$text) {
    $beforeKv = [double](([regex]::Match($text, "\[Before Clear\][\s\S]*?KV bytes:\s+([0-9.]+) MB")).Groups[1].Value)
    $beforeGpu = [double](([regex]::Match($text, "\[Before Clear\][\s\S]*?GPU used:\s+([0-9.]+) MB")).Groups[1].Value)
    $afterKv = [double](([regex]::Match($text, "\[After Clear\][\s\S]*?KV bytes:\s+([0-9.]+) MB")).Groups[1].Value)
    $afterGpu = [double](([regex]::Match($text, "\[After Clear\][\s\S]*?GPU used:\s+([0-9.]+) MB")).Groups[1].Value)
    $reclaimed = [double](([regex]::Match($text, "GPU reclaimed:\s+([0-9.]+) MB")).Groups[1].Value)
    return @{
        kv_before = $beforeKv
        kv_after = $afterKv
        gpu_before = $beforeGpu
        gpu_after = $afterGpu
        gpu_reclaimed = $reclaimed
    }
}

Ensure-BenchmarkBuild

Push-Location $repoRoot
try {
    $gpuInfo = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader) | Select-Object -First 1
    $nvccVersion = (& nvcc --version | Out-String).Trim()

    $kvOutput = (& $kvExe $KvDecodeSteps | Out-String)
    $batchOutput = (& $batchExe $BatchSteps | Out-String)
    $pagedOutput = (& $pagedExe $PagedMaxSeqLen $PagedPageSize $PagedMaxPages | Out-String)

    $kvCsvPath = Join-Path $repoRoot "kv_int8_benchmark_results.csv"
    $batchCsvPath = Join-Path $repoRoot "batch_poc_benchmark_results.csv"
    $pagedCsvPath = Join-Path $repoRoot "paged_kv_benchmark_results.csv"

    if (-not (Test-Path $kvCsvPath)) { throw "Missing kv_int8_benchmark_results.csv" }
    if (-not (Test-Path $batchCsvPath)) { throw "Missing batch_poc_benchmark_results.csv" }
    if (-not (Test-Path $pagedCsvPath)) { throw "Missing paged_kv_benchmark_results.csv" }

    $kvRows = Import-Csv $kvCsvPath
    $batchParsed = Parse-BatchOutput -text $batchOutput
    $pagedParsed = Parse-PagedOutput -text $pagedOutput
    $pagedRows = Import-Csv $pagedCsvPath | Where-Object { $_.seq_len -and $_.allocated_pages -and $_.kv_bytes_mb -and $_.gpu_used_mb }

    $seq0 = $pagedRows | Where-Object { [int]$_.seq_len -eq 0 } | Select-Object -First 1
    $seq1024 = $pagedRows | Where-Object { [int]$_.seq_len -eq 1024 } | Select-Object -First 1
    $seqMax = $pagedRows | Where-Object { [int]$_.seq_len -eq $PagedMaxSeqLen } | Select-Object -First 1

    $today = Get-Date -Format "yyyy-MM-dd"

    $fp32Avg = Get-ValueFromKvCsv $kvRows "avg_latency_ms" "fp32"
    $int8Avg = Get-ValueFromKvCsv $kvRows "avg_latency_ms" "int8"
    $fp32P50 = Get-ValueFromKvCsv $kvRows "p50_latency_ms" "fp32"
    $int8P50 = Get-ValueFromKvCsv $kvRows "p50_latency_ms" "int8"
    $fp32P95 = Get-ValueFromKvCsv $kvRows "p95_latency_ms" "fp32"
    $int8P95 = Get-ValueFromKvCsv $kvRows "p95_latency_ms" "int8"
    $fp32Tps = Get-ValueFromKvCsv $kvRows "throughput_tps" "fp32"
    $int8Tps = Get-ValueFromKvCsv $kvRows "throughput_tps" "int8"
    $fp32Vram = Get-ValueFromKvCsv $kvRows "vram_after_init_mb" "fp32"
    $int8Vram = Get-ValueFromKvCsv $kvRows "vram_after_init_mb" "int8"
    $fp32Peak = Get-ValueFromKvCsv $kvRows "peak_vram_mb" "fp32"
    $int8Peak = Get-ValueFromKvCsv $kvRows "peak_vram_mb" "int8"
    $fp32Alloc = Get-ValueFromKvCsv $kvRows "model_alloc_mb" "fp32"
    $int8Alloc = Get-ValueFromKvCsv $kvRows "model_alloc_mb" "int8"
    $int8MaxDiff = Get-ValueFromKvCsv $kvRows "max_abs_diff" "int8"
    $int8MeanDiff = Get-ValueFromKvCsv $kvRows "mean_abs_diff" "int8"
    $int8P99 = Get-ValueFromKvCsv $kvRows "p99_diff" "int8"
    $int8Match = (Get-ValueFromKvCsv $kvRows "token_match_rate" "int8") * 100.0

    $mdLines = @(
        "# Latest Benchmark Summary (Single Source of Truth)",
        "",
        "This document records the latest reproducible benchmark run for memory/performance optimization experiments.",
        "",
        "## 1) Test Environment",
        "",
        "- Date: $today",
        "- OS: Windows 10 (PowerShell)",
        "- GPU/Driver/VRAM: $gpuInfo",
        "- CUDA Toolkit (nvcc):",
        "  $nvccVersion",
        "- Build Mode: $BuildConfig",
        "- Build Command:",
        "  - cmake -B build -DCMAKE_BUILD_TYPE=$BuildConfig -DENABLE_CUDA=ON",
        "  - cmake --build build --config $BuildConfig --target kv_int8_benchmark batch_poc_benchmark paged_kv_benchmark",
        "",
        "## 2) Metric Definitions",
        "",
        "- avg latency (ms/token): arithmetic mean of per-step decode latency.",
        "- p50 latency (ms): median latency (50th percentile) across measured decode steps.",
        "- p95 latency (ms): 95th percentile latency across measured decode steps.",
        "- throughput (tok/s):",
        "  - Decode benchmark: 1000 / avg_latency_ms",
        "  - Batch benchmark (forward_batch): batch_size * 1000 / avg_step_ms",
        "",
        "## 3) Benchmark Commands and Sample Sizes",
        "",
        "### 3.1 KV INT8 A/B",
        "- Command: ./build/$BuildConfig/kv_int8_benchmark.exe $KvDecodeSteps",
        "- Samples: decode steps=$KvDecodeSteps, repeats=1",
        "",
        "### 3.2 Batch POC",
        "- Command: ./build/$BuildConfig/batch_poc_benchmark.exe $BatchSteps",
        "- Samples: batch sizes=1,2,4, steps=$BatchSteps, repeats=1",
        "",
        "### 3.3 Paged KV",
        "- Command: ./build/$BuildConfig/paged_kv_benchmark.exe $PagedMaxSeqLen $PagedPageSize $PagedMaxPages",
        "- Samples: seq scan points=0..$PagedMaxSeqLen (stride $PagedPageSize), clear check=1",
        "",
        "## 4) Latest Results",
        "",
        "### 4.1 KV INT8 A/B",
        "",
        "| Metric | FP32 Baseline | INT8 KV |",
        "|---|---:|---:|",
        "| Decode avg latency (ms/token) | $([string]::Format('{0:F3}', $fp32Avg)) | $([string]::Format('{0:F3}', $int8Avg)) |",
        "| Decode p50 (ms) | $([string]::Format('{0:F3}', $fp32P50)) | $([string]::Format('{0:F3}', $int8P50)) |",
        "| Decode p95 (ms) | $([string]::Format('{0:F3}', $fp32P95)) | $([string]::Format('{0:F3}', $int8P95)) |",
        "| Decode throughput (tok/s) | $([string]::Format('{0:F3}', $fp32Tps)) | $([string]::Format('{0:F3}', $int8Tps)) |",
        "| VRAM after init (MB) | $([string]::Format('{0:F0}', $fp32Vram)) | $([string]::Format('{0:F0}', $int8Vram)) |",
        "| Peak VRAM (MB) | $([string]::Format('{0:F0}', $fp32Peak)) | $([string]::Format('{0:F0}', $int8Peak)) |",
        "| Model allocation (MB) | $([string]::Format('{0:F0}', $fp32Alloc)) | $([string]::Format('{0:F0}', $int8Alloc)) |",
        "| Max abs diff | 0.000 | $([string]::Format('{0:F3}', $int8MaxDiff)) |",
        "| Mean abs diff | 0.000 | $([string]::Format('{0:F3}', $int8MeanDiff)) |",
        "| P99 diff | 0.000 | $([string]::Format('{0:F3}', $int8P99)) |",
        "| Top-1 token match rate | 100.000% | $([string]::Format('{0:F3}', $int8Match))% |",
        "",
        "### 4.2 Batch POC (forward_batch)",
        "",
        "| batch_size | avg_step (ms) | throughput (tok/s) | speedup vs batch=1 |",
        "|---:|---:|---:|---:|",
        "| 1 | $([string]::Format('{0:F3}', $batchParsed.forward[1].avg_step_ms)) | $([string]::Format('{0:F3}', $batchParsed.forward[1].throughput_tps)) | $([string]::Format('{0:F3}', $batchParsed.forward[1].speedup))x |",
        "| 2 | $([string]::Format('{0:F3}', $batchParsed.forward[2].avg_step_ms)) | $([string]::Format('{0:F3}', $batchParsed.forward[2].throughput_tps)) | $([string]::Format('{0:F3}', $batchParsed.forward[2].speedup))x |",
        "| 4 | $([string]::Format('{0:F3}', $batchParsed.forward[4].avg_step_ms)) | $([string]::Format('{0:F3}', $batchParsed.forward[4].throughput_tps)) | $([string]::Format('{0:F3}', $batchParsed.forward[4].speedup))x |",
        "",
        "### 4.3 Batch POC (sequential baseline)",
        "",
        "| batch_size | throughput (tok/s) |",
        "|---:|---:|",
        "| 1 | $([string]::Format('{0:F3}', $batchParsed.sequential[1])) |",
        "| 2 | $([string]::Format('{0:F3}', $batchParsed.sequential[2])) |",
        "| 4 | $([string]::Format('{0:F3}', $batchParsed.sequential[4])) |",
        "",
        "### 4.4 API Gain (forward_batch / sequential)",
        "",
        "| batch_size | gain |",
        "|---:|---:|",
        "| 1 | $([string]::Format('{0:F3}', $batchParsed.gain[1]))x |",
        "| 2 | $([string]::Format('{0:F3}', $batchParsed.gain[2]))x |",
        "| 4 | $([string]::Format('{0:F3}', $batchParsed.gain[4]))x |",
        "",
        "### 4.5 Paged KV Scan and Reclaim",
        "",
        "Key points from seq_len scan:",
        "- seq_len=0: pages=$($seq0.allocated_pages), kv_bytes=$($seq0.kv_bytes_mb) MB, gpu_used=$($seq0.gpu_used_mb) MB",
        "- seq_len=1024: pages=$($seq1024.allocated_pages), kv_bytes=$($seq1024.kv_bytes_mb) MB, gpu_used=$($seq1024.gpu_used_mb) MB",
        "- seq_len=${PagedMaxSeqLen}: pages=$($seqMax.allocated_pages), kv_bytes=$($seqMax.kv_bytes_mb) MB, gpu_used=$($seqMax.gpu_used_mb) MB",
        "",
        "Clear/reclaim:",
        "",
        "| Metric | Before Clear | After Clear | Delta |",
        "|---|---:|---:|---:|",
        "| kv_bytes_mb | $([string]::Format('{0:F2}', $pagedParsed.kv_before)) | $([string]::Format('{0:F2}', $pagedParsed.kv_after)) | $([string]::Format('{0:F2}', ($pagedParsed.kv_after - $pagedParsed.kv_before))) |",
        "| gpu_used_mb | $([string]::Format('{0:F2}', $pagedParsed.gpu_before)) | $([string]::Format('{0:F2}', $pagedParsed.gpu_after)) | $([string]::Format('{0:F2}', ($pagedParsed.gpu_after - $pagedParsed.gpu_before))) |",
        "| gpu_reclaimed_mb | - | - | $([string]::Format('{0:F2}', $pagedParsed.gpu_reclaimed)) |",
        "",
        "## 5) Raw Artifacts",
        "",
        "- kv_int8_benchmark_results.csv",
        "- batch_poc_benchmark_results.csv",
        "- paged_kv_benchmark_results.csv"
    )
    $md = ($mdLines -join "`r`n")

    Set-Content -Path $summaryMd -Value $md -Encoding UTF8

    @(
        "group,metric,key,value"
        "kv_int8,latency,fp32_avg_ms_per_token,$fp32Avg"
        "kv_int8,latency,int8_avg_ms_per_token,$int8Avg"
        "kv_int8,throughput,fp32_tps,$fp32Tps"
        "kv_int8,throughput,int8_tps,$int8Tps"
        "kv_int8,memory,fp32_model_alloc_mb,$fp32Alloc"
        "kv_int8,memory,int8_model_alloc_mb,$int8Alloc"
        "batch_poc,forward_batch,b1_throughput_tps,$($batchParsed.forward[1].throughput_tps)"
        "batch_poc,forward_batch,b2_throughput_tps,$($batchParsed.forward[2].throughput_tps)"
        "batch_poc,forward_batch,b4_throughput_tps,$($batchParsed.forward[4].throughput_tps)"
        "batch_poc,api_gain,b2_gain,$($batchParsed.gain[2])"
        "batch_poc,api_gain,b4_gain,$($batchParsed.gain[4])"
        "paged_kv,reclaim,kv_before_mb,$($pagedParsed.kv_before)"
        "paged_kv,reclaim,kv_after_mb,$($pagedParsed.kv_after)"
        "paged_kv,reclaim,gpu_reclaimed_mb,$($pagedParsed.gpu_reclaimed)"
    ) | Set-Content -Path $summaryCsv -Encoding UTF8

    Write-Host "Benchmark suite completed."
    Write-Host "Generated:"
    Write-Host "  - $summaryMd"
    Write-Host "  - $summaryCsv"
    Write-Host "  - $kvCsvPath"
    Write-Host "  - $batchCsvPath"
    Write-Host "  - $pagedCsvPath"
} finally {
    Pop-Location
}
