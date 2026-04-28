#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <mutex>
#include <functional>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace qwen {
namespace cuda {

// ============================================================================
// GPU Memory Profiler - 显存分析工具
// ============================================================================
// 功能：
// 1. 追踪所有 cudaMalloc/cudaFree 调用
// 2. 按模块分类统计内存占用
// 3. 在运行时打印内存快照
// 4. 检测内存泄漏
// ============================================================================

struct MemoryBlock {
    void* ptr;
    size_t size;
    std::string tag;        // 内存用途标签，如 "kv_cache", "weights", "activation"
    std::string module;     // 所属模块，如 "CudaEngine", "CudaFullAttention"
    std::string file;
    int line;
    uint64_t timestamp_us;
};

class GPUMemoryProfiler {
public:
    static GPUMemoryProfiler& instance() {
        static GPUMemoryProfiler profiler;
        return profiler;
    }

    // 注册一次内存分配
    void record_alloc(void* ptr, size_t size, const char* tag, const char* module,
                      const char* file, int line);

    // 注册一次内存释放
    void record_free(void* ptr);

    // 打印当前内存快照
    void print_snapshot(const std::string& label = "") const;

    // 打印按模块汇总的内存统计
    void print_module_summary() const;

    // 打印按标签汇总的内存统计
    void print_tag_summary() const;

    // 获取当前总分配内存
    size_t total_allocated_bytes() const;

    // 获取当前活跃内存块数量
    size_t active_block_count() const;

    // 检查内存泄漏（应在程序退出前调用）
    void check_leaks() const;

    // 重置所有记录
    void reset();

    // 获取 CUDA 实际显存使用
    static void print_cuda_memory_info(const std::string& label = "");

    // 带标签的封装分配函数
    static cudaError_t malloc_tagged(void** ptr, size_t size, const char* tag,
                                     const char* module, const char* file, int line);

private:
    GPUMemoryProfiler() = default;
    ~GPUMemoryProfiler() { check_leaks(); }

    mutable std::mutex mutex_;
    std::unordered_map<void*, MemoryBlock> active_blocks_;
    std::vector<MemoryBlock> freed_history_;  // 可选：记录已释放的内存历史

    uint64_t get_timestamp_us() const;
};

// ============================================================================
// 便捷宏定义
// ============================================================================

#define GPU_MALLOC_TAGGED(ptr, size, tag, module) \
    qwen::cuda::GPUMemoryProfiler::malloc_tagged( \
        reinterpret_cast<void**>(&ptr), size, tag, module, __FILE__, __LINE__)

#define GPU_RECORD_ALLOC(ptr, size, tag, module) \
    qwen::cuda::GPUMemoryProfiler::instance().record_alloc( \
        ptr, size, tag, module, __FILE__, __LINE__)

#define GPU_RECORD_FREE(ptr) \
    qwen::cuda::GPUMemoryProfiler::instance().record_free(ptr)

#define GPU_MEM_SNAPSHOT(label) \
    qwen::cuda::GPUMemoryProfiler::instance().print_snapshot(label); \
    qwen::cuda::GPUMemoryProfiler::print_cuda_memory_info(label)

#define GPU_MEM_MODULE_SUMMARY() \
    qwen::cuda::GPUMemoryProfiler::instance().print_module_summary()

#define GPU_MEM_TAG_SUMMARY() \
    qwen::cuda::GPUMemoryProfiler::instance().print_tag_summary()

// ============================================================================
// ScopedMemorySnapshot - 作用域内存快照
// ============================================================================
// 在构造和析构时自动打印内存变化

class ScopedMemorySnapshot {
public:
    explicit ScopedMemorySnapshot(const std::string& scope_name);
    ~ScopedMemorySnapshot();

private:
    std::string scope_name_;
    size_t start_allocated_;
    size_t start_cuda_used_;
};

#define SCOPED_MEM_SNAPSHOT(name) \
    qwen::cuda::ScopedMemorySnapshot _mem_snap_##__LINE__(name)

} // namespace cuda
} // namespace qwen
