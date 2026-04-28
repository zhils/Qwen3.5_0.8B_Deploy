#include "gpu_memory_profiler.hpp"
#include <chrono>
#include <algorithm>
#include <numeric>

namespace qwen {
namespace cuda {

// ============================================================================
// GPUMemoryProfiler Implementation
// ============================================================================

void GPUMemoryProfiler::record_alloc(void* ptr, size_t size, const char* tag,
                                     const char* module, const char* file, int line) {
    if (!ptr || size == 0) return;

    std::lock_guard<std::mutex> lock(mutex_);

    MemoryBlock block;
    block.ptr = ptr;
    block.size = size;
    block.tag = tag ? tag : "untagged";
    block.module = module ? module : "unknown";
    block.file = file ? file : "";
    block.line = line;
    block.timestamp_us = get_timestamp_us();

    active_blocks_[ptr] = block;
}

void GPUMemoryProfiler::record_free(void* ptr) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = active_blocks_.find(ptr);
    if (it != active_blocks_.end()) {
        freed_history_.push_back(it->second);
        active_blocks_.erase(it);
    }
}

size_t GPUMemoryProfiler::total_allocated_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto& [ptr, block] : active_blocks_) {
        total += block.size;
    }
    return total;
}

size_t GPUMemoryProfiler::active_block_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_blocks_.size();
}

void GPUMemoryProfiler::print_snapshot(const std::string& label) const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total_bytes = 0;
    for (const auto& [ptr, block] : active_blocks_) {
        total_bytes += block.size;
    }

    std::cout << "\n" << std::string(72, '=') << std::endl;
    if (!label.empty()) {
        std::cout << "  GPU Memory Snapshot: " << label << std::endl;
    } else {
        std::cout << "  GPU Memory Snapshot" << std::endl;
    }
    std::cout << std::string(72, '=') << std::endl;
    std::cout << "  Active blocks: " << active_blocks_.size() << std::endl;
    std::cout << "  Total allocated: " << (total_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;

    if (!active_blocks_.empty()) {
        std::cout << "\n  Top 10 largest blocks:" << std::endl;
        std::vector<MemoryBlock> blocks;
        for (const auto& [ptr, block] : active_blocks_) {
            blocks.push_back(block);
        }
        std::sort(blocks.begin(), blocks.end(),
                  [](const MemoryBlock& a, const MemoryBlock& b) { return a.size > b.size; });

        std::cout << "  " << std::left << std::setw(18) << "Tag"
                  << std::setw(18) << "Module"
                  << std::setw(14) << "Size (MB)"
                  << std::setw(16) << "Ptr" << std::endl;
        std::cout << "  " << std::string(66, '-') << std::endl;

        for (size_t i = 0; i < std::min(blocks.size(), size_t(10)); ++i) {
            const auto& b = blocks[i];
            std::cout << "  " << std::left << std::setw(18) << b.tag.substr(0, 17)
                      << std::setw(18) << b.module.substr(0, 17)
                      << std::setw(14) << (b.size / (1024.0 * 1024.0))
                      << std::setw(16) << b.ptr << std::endl;
        }
    }
    std::cout << std::string(72, '=') << std::endl;
}

void GPUMemoryProfiler::print_module_summary() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::unordered_map<std::string, size_t> module_bytes;
    std::unordered_map<std::string, size_t> module_count;

    for (const auto& [ptr, block] : active_blocks_) {
        module_bytes[block.module] += block.size;
        module_count[block.module]++;
    }

    std::vector<std::pair<std::string, size_t>> sorted_modules;
    for (const auto& [mod, bytes] : module_bytes) {
        sorted_modules.push_back({mod, bytes});
    }
    std::sort(sorted_modules.begin(), sorted_modules.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::cout << "\n" << std::string(72, '=') << std::endl;
    std::cout << "  GPU Memory by Module" << std::endl;
    std::cout << std::string(72, '=') << std::endl;
    std::cout << "  " << std::left << std::setw(24) << "Module"
              << std::setw(14) << "Blocks"
              << std::setw(14) << "Size (MB)"
              << std::setw(10) << "Percent" << std::endl;
    std::cout << "  " << std::string(62, '-') << std::endl;

    size_t total = 0;
    for (const auto& [mod, bytes] : sorted_modules) {
        total += bytes;
    }

    for (const auto& [mod, bytes] : sorted_modules) {
        float pct = total > 0 ? (100.0f * bytes / total) : 0.0f;
        std::cout << "  " << std::left << std::setw(24) << mod.substr(0, 23)
                  << std::setw(14) << module_count[mod]
                  << std::setw(14) << (bytes / (1024.0 * 1024.0))
                  << std::setw(10) << std::fixed << std::setprecision(1) << pct << "%" << std::endl;
    }
    std::cout << "  " << std::string(62, '-') << std::endl;
    std::cout << "  " << std::left << std::setw(24) << "TOTAL"
              << std::setw(14) << active_blocks_.size()
              << std::setw(14) << (total / (1024.0 * 1024.0)) << std::endl;
    std::cout << std::string(72, '=') << std::endl;
}

void GPUMemoryProfiler::print_tag_summary() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::unordered_map<std::string, size_t> tag_bytes;
    std::unordered_map<std::string, size_t> tag_count;

    for (const auto& [ptr, block] : active_blocks_) {
        tag_bytes[block.tag] += block.size;
        tag_count[block.tag]++;
    }

    std::vector<std::pair<std::string, size_t>> sorted_tags;
    for (const auto& [tag, bytes] : tag_bytes) {
        sorted_tags.push_back({tag, bytes});
    }
    std::sort(sorted_tags.begin(), sorted_tags.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::cout << "\n" << std::string(72, '=') << std::endl;
    std::cout << "  GPU Memory by Tag" << std::endl;
    std::cout << std::string(72, '=') << std::endl;
    std::cout << "  " << std::left << std::setw(24) << "Tag"
              << std::setw(14) << "Blocks"
              << std::setw(14) << "Size (MB)"
              << std::setw(10) << "Percent" << std::endl;
    std::cout << "  " << std::string(62, '-') << std::endl;

    size_t total = 0;
    for (const auto& [tag, bytes] : sorted_tags) {
        total += bytes;
    }

    for (const auto& [tag, bytes] : sorted_tags) {
        float pct = total > 0 ? (100.0f * bytes / total) : 0.0f;
        std::cout << "  " << std::left << std::setw(24) << tag.substr(0, 23)
                  << std::setw(14) << tag_count[tag]
                  << std::setw(14) << (bytes / (1024.0 * 1024.0))
                  << std::setw(10) << std::fixed << std::setprecision(1) << pct << "%" << std::endl;
    }
    std::cout << "  " << std::string(62, '-') << std::endl;
    std::cout << "  " << std::left << std::setw(24) << "TOTAL"
              << std::setw(14) << active_blocks_.size()
              << std::setw(14) << (total / (1024.0 * 1024.0)) << std::endl;
    std::cout << std::string(72, '=') << std::endl;
}

void GPUMemoryProfiler::check_leaks() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (active_blocks_.empty()) return;

    std::cerr << "\n" << std::string(72, '!') << std::endl;
    std::cerr << "  WARNING: " << active_blocks_.size()
              << " GPU memory blocks not freed (potential leaks)" << std::endl;
    std::cerr << std::string(72, '!') << std::endl;

    for (const auto& [ptr, block] : active_blocks_) {
        std::cerr << "  Leak: " << block.size << " bytes at " << ptr
                  << " [" << block.module << "/" << block.tag << "]"
                  << " allocated at " << block.file << ":" << block.line << std::endl;
    }
}

void GPUMemoryProfiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    active_blocks_.clear();
    freed_history_.clear();
}

void GPUMemoryProfiler::print_cuda_memory_info(const std::string& label) {
    size_t free_mem = 0, total_mem = 0;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        std::cerr << "  Failed to get CUDA memory info: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    size_t used_mem = total_mem - free_mem;

    std::cout << "  CUDA Memory Info";
    if (!label.empty()) std::cout << " [" << label << "]";
    std::cout << ":" << std::endl;
    std::cout << "    Free:  " << (free_mem / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "    Used:  " << (used_mem / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "    Total: " << (total_mem / (1024.0 * 1024.0)) << " MB" << std::endl;
}

cudaError_t GPUMemoryProfiler::malloc_tagged(void** ptr, size_t size, const char* tag,
                                             const char* module, const char* file, int line) {
    cudaError_t err = cudaMalloc(ptr, size);
    if (err == cudaSuccess && *ptr) {
        instance().record_alloc(*ptr, size, tag, module, file, line);
    }
    return err;
}

uint64_t GPUMemoryProfiler::get_timestamp_us() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

// ============================================================================
// ScopedMemorySnapshot Implementation
// ============================================================================

ScopedMemorySnapshot::ScopedMemorySnapshot(const std::string& scope_name)
    : scope_name_(scope_name) {
    start_allocated_ = GPUMemoryProfiler::instance().total_allocated_bytes();

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    start_cuda_used_ = total_mem - free_mem;

    std::cout << "\n  [MEM] Enter scope: " << scope_name_ << std::endl;
    std::cout << "        Tracked: " << (start_allocated_ / (1024.0 * 1024.0))
              << " MB | CUDA used: " << (start_cuda_used_ / (1024.0 * 1024.0)) << " MB" << std::endl;
}

ScopedMemorySnapshot::~ScopedMemorySnapshot() {
    size_t end_allocated = GPUMemoryProfiler::instance().total_allocated_bytes();

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t end_cuda_used = total_mem - free_mem;

    int64_t delta_tracked = static_cast<int64_t>(end_allocated) - static_cast<int64_t>(start_allocated_);
    int64_t delta_cuda = static_cast<int64_t>(end_cuda_used) - static_cast<int64_t>(start_cuda_used_);

    std::cout << "  [MEM] Exit scope: " << scope_name_ << std::endl;
    std::cout << "        Tracked: " << (end_allocated / (1024.0 * 1024.0)) << " MB ("
              << (delta_tracked >= 0 ? "+" : "") << (delta_tracked / (1024.0 * 1024.0)) << " MB)" << std::endl;
    std::cout << "        CUDA used: " << (end_cuda_used / (1024.0 * 1024.0)) << " MB ("
              << (delta_cuda >= 0 ? "+" : "") << (delta_cuda / (1024.0 * 1024.0)) << " MB)" << std::endl;
}

} // namespace cuda
} // namespace qwen
