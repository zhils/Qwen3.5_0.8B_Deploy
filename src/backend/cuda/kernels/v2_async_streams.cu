/**
 * v2.0 Async Data Transfer & Multi-Stream Parallelism
 *
 * Features:
 * 1. cudaMemcpyAsync for overlapping H2D/D2H with computation
 * 2. Multiple CUDA streams for parallel kernel execution
 * 3. Stream synchronization primitives
 *
 * Stream assignment strategy:
 * - Stream 0 (default): Attention computation
 * - Stream 1: MLP computation  
 * - Stream 2: Data transfer (H2D/D2H)
 */

#include "cuda_error_handling.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

namespace qwen {
namespace cuda {

// ============================================================================
// Multi-Stream Manager
// ============================================================================

class StreamManager {
public:
    static constexpr int NUM_STREAMS = 3;

    enum StreamId {
        COMPUTE_ATTN = 0,  // Attention and normalization
        COMPUTE_MLP = 1,   // MLP and projections
        DATA_TRANSFER = 2, // H2D/D2H transfers
    };

    StreamManager();
    ~StreamManager();

    cudaStream_t get(StreamId id) const {
        return streams_[static_cast<int>(id)];
    }

    cudaStream_t operator[](int idx) const {
        return streams_[idx];
    }

    void synchronize(StreamId id) const;
    void synchronize_all() const;

    // Record event on a stream
    void record_event(StreamId id, cudaEvent_t event);

    // Wait for event on a stream
    void wait_event(StreamId id, cudaEvent_t event);

    // Create events for synchronization
    void create_events(int count);
    void destroy_events();
    cudaEvent_t get_event(int idx) const;

private:
    cudaStream_t streams_[NUM_STREAMS];
    std::vector<cudaEvent_t> events_;
    bool initialized_;
};

StreamManager::StreamManager() : initialized_(false) {
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaError_t err = cudaStreamCreate(&streams_[i]);
        if (err != cudaSuccess) {
            // Clean up already created streams
            for (int j = 0; j < i; ++j) {
                cudaStreamDestroy(streams_[j]);
            }
            throw std::runtime_error("Failed to create CUDA stream " + std::to_string(i));
        }
    }
    initialized_ = true;
}

StreamManager::~StreamManager() {
    if (initialized_) {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            if (streams_[i]) {
                cudaStreamDestroy(streams_[i]);
            }
        }
        destroy_events();
    }
}

void StreamManager::synchronize(StreamId id) const {
    cudaStreamSynchronize(streams_[static_cast<int>(id)]);
}

void StreamManager::synchronize_all() const {
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams_[i]);
    }
}

void StreamManager::create_events(int count) {
    destroy_events();
    events_.resize(count);
    for (int i = 0; i < count; ++i) {
        cudaEventCreate(&events_[i]);
    }
}

void StreamManager::destroy_events() {
    for (auto& e : events_) {
        if (e) {
            cudaEventDestroy(e);
        }
    }
    events_.clear();
}

cudaEvent_t StreamManager::get_event(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(events_.size())) {
        return nullptr;
    }
    return events_[idx];
}

void StreamManager::record_event(StreamId id, cudaEvent_t event) {
    cudaEventRecord(event, streams_[static_cast<int>(id)]);
}

void StreamManager::wait_event(StreamId id, cudaEvent_t event) {
    cudaStreamWaitEvent(streams_[static_cast<int>(id)], event, 0);
}

// ============================================================================
// Async Buffer Manager
// ============================================================================

class AsyncBufferManager {
public:
    AsyncBufferManager();
    ~AsyncBufferManager();

    // Allocate pinned host buffer for async transfers
    void allocate_host_buffer(size_t bytes);
    void free_host_buffer();

    // Async H2D copy
    void copy_h2d_async(void* d_dst, const void* h_src, size_t bytes, cudaStream_t stream);

    // Async D2H copy
    void copy_d2h_async(void* h_dst, const void* d_src, size_t bytes, cudaStream_t stream);

    // Async D2D copy
    void copy_d2d_async(void* d_dst, const void* d_src, size_t bytes, cudaStream_t stream);

    // Get pinned host buffer
    void* host_buffer() const { return h_pinned_buf_; }
    size_t host_buffer_size() const { return host_buf_size_; }

private:
    void* h_pinned_buf_;
    size_t host_buf_size_;
};

AsyncBufferManager::AsyncBufferManager() : h_pinned_buf_(nullptr), host_buf_size_(0) {}

AsyncBufferManager::~AsyncBufferManager() {
    free_host_buffer();
}

void AsyncBufferManager::allocate_host_buffer(size_t bytes) {
    if (h_pinned_buf_ && host_buf_size_ >= bytes) {
        return; // Already have enough
    }
    free_host_buffer();
    cudaMallocHost(&h_pinned_buf_, bytes);
    host_buf_size_ = bytes;
}

void AsyncBufferManager::free_host_buffer() {
    if (h_pinned_buf_) {
        cudaFreeHost(h_pinned_buf_);
        h_pinned_buf_ = nullptr;
        host_buf_size_ = 0;
    }
}

void AsyncBufferManager::copy_h2d_async(void* d_dst, const void* h_src, size_t bytes,
                                        cudaStream_t stream) {
    cudaMemcpyAsync(d_dst, h_src, bytes, cudaMemcpyHostToDevice, stream);
}

void AsyncBufferManager::copy_d2h_async(void* h_dst, const void* d_src, size_t bytes,
                                        cudaStream_t stream) {
    cudaMemcpyAsync(h_dst, d_src, bytes, cudaMemcpyDeviceToHost, stream);
}

void AsyncBufferManager::copy_d2d_async(void* d_dst, const void* d_src, size_t bytes,
                                        cudaStream_t stream) {
    cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice, stream);
}

// ============================================================================
// v2.0 CUDA Engine with all optimizations
// ============================================================================

// Forward declarations for v2 kernels
void flash_attention_v2_decode(const float* d_Q, const float* d_K_cache, const float* d_V_cache,
                               float* d_output, int num_heads, int num_kv_heads, int q_head_dim,
                               int kv_head_dim, int seq_len, float scale);

} // namespace cuda
} // namespace qwen
