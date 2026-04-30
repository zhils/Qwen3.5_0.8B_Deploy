#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <unordered_map>

namespace qwen {
namespace cuda {

struct KVPage {
    int page_id;
    int ref_count;
    int seq_len;
    bool in_use;
};

class PagedKVCache {
  public:
    PagedKVCache();
    ~PagedKVCache();

    void initialize(int num_layers, int num_kv_heads, int head_dim, int page_size, int max_pages);
    void clear();

    int allocate_sequence(int initial_len = 0);
    void append_tokens(int seq_id, int num_tokens);
    void release_sequence(int seq_id);
    void shrink_sequence(int seq_id, int new_len);

    size_t get_kv_bytes() const;
    size_t get_gpu_bytes() const;
    int get_allocated_pages() const;
    int get_used_pages() const;

    void print_status() const;

  private:
    int allocate_page();
    void release_page(int page_id);

    float* d_kv_data_;
    int* d_page_tables_;

    std::vector<KVPage> pages_;
    std::unordered_map<int, std::vector<int>> seq_to_pages_;
    std::unordered_map<int, int> seq_lengths_;

    int num_layers_;
    int num_kv_heads_;
    int head_dim_;
    int page_size_;
    int max_pages_;
    int next_seq_id_;

    size_t kv_bytes_;
    size_t gpu_bytes_;
};

} // namespace cuda
} // namespace qwen
