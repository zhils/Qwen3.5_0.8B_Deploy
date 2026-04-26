/**
 * v2.0 Paged KV Cache Implementation
 *
 * Instead of pre-allocating max_seq_len for all layers, allocate pages on demand.
 * This reduces memory footprint for short sequences and enables longer sequences.
 *
 * Page size: 128 tokens (configurable)
 * Each page is [page_size, num_kv_heads, head_dim] contiguous in memory
 */

#include "cuda_error_handling.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>

namespace qwen {
namespace cuda {

// ============================================================================
// Paged KV Cache Structure
// ============================================================================

static constexpr int PAGE_SIZE = 128; // tokens per page

struct PagedKVCache {
    // Page table: layer -> list of page indices
    // page_table[layer][page_idx] = physical_page_id
    std::vector<std::vector<int>> page_table;

    // Physical pages: flat array of [num_physical_pages][PAGE_SIZE][num_kv_heads][head_dim]
    float* d_k_pages;
    float* d_v_pages;
    int num_physical_pages;
    int num_layers;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;

    // Free page list
    std::vector<int> free_pages;

    // Per-layer sequence lengths
    std::vector<int> layer_seq_lengths;

    void init(int nl, int nkh, int hd, int max_len) {
        num_layers = nl;
        num_kv_heads = nkh;
        head_dim = hd;
        max_seq_len = max_len;

        // Calculate max pages needed
        int max_pages_per_layer = (max_len + PAGE_SIZE - 1) / PAGE_SIZE;
        int total_pages = nl * max_pages_per_layer;

        size_t page_bytes = static_cast<size_t>(PAGE_SIZE) * nkh * hd * sizeof(float);
        cudaMalloc(&d_k_pages, total_pages * page_bytes);
        cudaMalloc(&d_v_pages, total_pages * page_bytes);
        cudaMemset(d_k_pages, 0, total_pages * page_bytes);
        cudaMemset(d_v_pages, 0, total_pages * page_bytes);

        num_physical_pages = total_pages;

        // Initialize free list
        free_pages.reserve(total_pages);
        for (int i = 0; i < total_pages; ++i) {
            free_pages.push_back(i);
        }

        page_table.resize(nl);
        layer_seq_lengths.assign(nl, 0);
    }

    void clear() {
        // Return all pages to free list
        free_pages.clear();
        for (int i = 0; i < num_physical_pages; ++i) {
            free_pages.push_back(i);
        }

        for (auto& pt : page_table) {
            pt.clear();
        }

        std::fill(layer_seq_lengths.begin(), layer_seq_lengths.end(), 0);

        size_t page_bytes = static_cast<size_t>(PAGE_SIZE) * num_kv_heads * head_dim * sizeof(float);
        cudaMemset(d_k_pages, 0, num_physical_pages * page_bytes);
        cudaMemset(d_v_pages, 0, num_physical_pages * page_bytes);
    }

    // Allocate a new page for a layer
    int allocate_page(int layer_idx) {
        if (free_pages.empty()) {
            fprintf(stderr, "PagedKVCache: out of pages!\n");
            return -1;
        }
        int page_id = free_pages.back();
        free_pages.pop_back();
        page_table[layer_idx].push_back(page_id);
        return page_id;
    }

    // Get physical address for a token position
    __host__ __device__ size_t get_offset(int layer_idx, int position) const {
        int page_idx = position / PAGE_SIZE;
        int offset_in_page = position % PAGE_SIZE;

        if (page_idx >= static_cast<int>(page_table[layer_idx].size())) {
            return 0; // Should not happen if properly allocated
        }

        int physical_page = page_table[layer_idx][page_idx];
        size_t page_stride = static_cast<size_t>(PAGE_SIZE) * num_kv_heads * head_dim;

        return static_cast<size_t>(physical_page) * page_stride +
               static_cast<size_t>(offset_in_page) * num_kv_heads * head_dim;
    }

    // Get K/V cache pointers for a specific layer and position
    __host__ float* get_k_ptr(int layer_idx, int position) {
        return d_k_pages + get_offset(layer_idx, position);
    }

    __host__ float* get_v_ptr(int layer_idx, int position) {
        return d_v_pages + get_offset(layer_idx, position);
    }

    int length(int layer_idx) const {
        return layer_seq_lengths[layer_idx];
    }

    size_t memory_used() const {
        size_t used_pages = 0;
        for (const auto& pt : page_table) {
            used_pages += pt.size();
        }
        return used_pages * PAGE_SIZE * num_kv_heads * head_dim * sizeof(float) * 2;
    }

    size_t memory_total() const {
        return static_cast<size_t>(num_physical_pages) * PAGE_SIZE * num_kv_heads * head_dim *
               sizeof(float) * 2;
    }
};

// ============================================================================
// Kernel: Write KV to paged cache
// ============================================================================

__global__ void write_kv_paged_kernel(const float* __restrict__ k_val,
                                      const float* __restrict__ v_val,
                                      PagedKVCache cache, int layer_idx, int position) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (h >= cache.num_kv_heads || d >= cache.head_dim)
        return;

    size_t offset = cache.get_offset(layer_idx, position) + h * cache.head_dim + d;
    int idx = h * cache.head_dim + d;

    cache.d_k_pages[offset] = k_val[idx];
    cache.d_v_pages[offset] = v_val[idx];
}

// ============================================================================
// Kernel: Read K cache tile for attention (coalesced access)
// ============================================================================

__global__ void load_k_tile_paged_kernel(const PagedKVCache cache, int layer_idx, int tile_start,
                                         int tile_len, float* __restrict__ s_k, int num_heads,
                                         int kv_head) {
    int tid = threadIdx.x;
    int head_dim = cache.head_dim;
    int num_kv_heads = cache.num_kv_heads;

    for (int i = tid; i < tile_len * head_dim; i += blockDim.x) {
        int t = i / head_dim;
        int d = i % head_dim;
        int seq_idx = tile_start + t;

        size_t offset = cache.get_offset(layer_idx, seq_idx) + kv_head * head_dim + d;
        s_k[t * head_dim + d] = cache.d_k_pages[offset];
    }
}

__global__ void load_v_tile_paged_kernel(const PagedKVCache cache, int layer_idx, int tile_start,
                                         int tile_len, float* __restrict__ s_v, int num_heads,
                                         int kv_head) {
    int tid = threadIdx.x;
    int head_dim = cache.head_dim;
    int num_kv_heads = cache.num_kv_heads;

    for (int i = tid; i < tile_len * head_dim; i += blockDim.x) {
        int t = i / head_dim;
        int d = i % head_dim;
        int seq_idx = tile_start + t;

        size_t offset = cache.get_offset(layer_idx, seq_idx) + kv_head * head_dim + d;
        s_v[t * head_dim + d] = cache.d_v_pages[offset];
    }
}

} // namespace cuda
} // namespace qwen
