#include "paged_kv.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

namespace qwen {
namespace cuda {

PagedKVCache::PagedKVCache()
    : d_kv_data_(nullptr), d_page_tables_(nullptr), num_layers_(0), num_kv_heads_(0), head_dim_(0),
      page_size_(0), max_pages_(0), next_seq_id_(1), kv_bytes_(0), gpu_bytes_(0) {}

PagedKVCache::~PagedKVCache() {
    clear();
}

void PagedKVCache::initialize(int num_layers, int num_kv_heads, int head_dim, int page_size,
                              int max_pages) {
    clear();

    num_layers_ = num_layers;
    num_kv_heads_ = num_kv_heads;
    head_dim_ = head_dim;
    page_size_ = page_size;
    max_pages_ = max_pages;

    pages_.resize(max_pages_);
    for (int i = 0; i < max_pages_; ++i) {
        pages_[i].page_id = i;
        pages_[i].ref_count = 0;
        pages_[i].seq_len = 0;
        pages_[i].in_use = false;
    }

    size_t kv_size_per_page =
        static_cast<size_t>(num_layers_) * 2 * num_kv_heads_ * head_dim_ * page_size_;
    size_t total_kv_size = kv_size_per_page * max_pages_;

    cudaMalloc(&d_kv_data_, total_kv_size * sizeof(float));
    cudaMalloc(&d_page_tables_, max_pages_ * sizeof(int));

    kv_bytes_ = total_kv_size * sizeof(float);
    gpu_bytes_ = kv_bytes_ + max_pages_ * sizeof(int);

    std::cout << "[PagedKV] Initialized: " << max_pages_ << " pages, "
              << (kv_bytes_ / (1024 * 1024)) << " MB KV cache" << std::endl;
}

void PagedKVCache::clear() {
    if (d_kv_data_) {
        cudaFree(d_kv_data_);
        d_kv_data_ = nullptr;
    }
    if (d_page_tables_) {
        cudaFree(d_page_tables_);
        d_page_tables_ = nullptr;
    }

    pages_.clear();
    seq_to_pages_.clear();
    seq_lengths_.clear();
    kv_bytes_ = 0;
    gpu_bytes_ = 0;
}

int PagedKVCache::allocate_page() {
    for (auto& page : pages_) {
        if (!page.in_use) {
            page.in_use = true;
            page.ref_count = 1;
            page.seq_len = 0;
            return page.page_id;
        }
    }
    return -1;
}

void PagedKVCache::release_page(int page_id) {
    if (page_id < 0 || page_id >= max_pages_)
        return;

    auto& page = pages_[page_id];
    page.ref_count--;
    if (page.ref_count <= 0) {
        page.in_use = false;
        page.seq_len = 0;
    }
}

int PagedKVCache::allocate_sequence(int initial_len) {
    int seq_id = next_seq_id_++;

    int num_pages_needed = (initial_len + page_size_ - 1) / page_size_;
    if (num_pages_needed == 0)
        num_pages_needed = 1;

    std::vector<int> page_ids;
    for (int i = 0; i < num_pages_needed; ++i) {
        int page_id = allocate_page();
        if (page_id < 0) {
            for (int pid : page_ids)
                release_page(pid);
            return -1;
        }
        page_ids.push_back(page_id);
    }

    seq_to_pages_[seq_id] = page_ids;
    seq_lengths_[seq_id] = initial_len;

    return seq_id;
}

void PagedKVCache::append_tokens(int seq_id, int num_tokens) {
    if (seq_to_pages_.find(seq_id) == seq_to_pages_.end())
        return;

    int current_len = seq_lengths_[seq_id];
    int new_len = current_len + num_tokens;

    int current_pages = (current_len + page_size_ - 1) / page_size_;
    int needed_pages = (new_len + page_size_ - 1) / page_size_;

    auto& page_ids = seq_to_pages_[seq_id];

    while (static_cast<int>(page_ids.size()) < needed_pages) {
        int new_page = allocate_page();
        if (new_page < 0) {
            std::cerr << "[PagedKV] Failed to allocate page for seq " << seq_id << std::endl;
            return;
        }
        page_ids.push_back(new_page);
    }

    seq_lengths_[seq_id] = new_len;
}

void PagedKVCache::release_sequence(int seq_id) {
    if (seq_to_pages_.find(seq_id) == seq_to_pages_.end())
        return;

    for (int page_id : seq_to_pages_[seq_id]) {
        release_page(page_id);
    }

    seq_to_pages_.erase(seq_id);
    seq_lengths_.erase(seq_id);
}

void PagedKVCache::shrink_sequence(int seq_id, int new_len) {
    if (seq_to_pages_.find(seq_id) == seq_to_pages_.end())
        return;
    if (new_len >= seq_lengths_[seq_id])
        return;

    auto& page_ids = seq_to_pages_[seq_id];
    int needed_pages = (new_len + page_size_ - 1) / page_size_;
    if (needed_pages == 0)
        needed_pages = 1;

    while (static_cast<int>(page_ids.size()) > needed_pages) {
        int page_id = page_ids.back();
        page_ids.pop_back();
        release_page(page_id);
    }

    seq_lengths_[seq_id] = new_len;
}

size_t PagedKVCache::get_kv_bytes() const {
    int used = get_used_pages();
    size_t kv_size_per_page =
        static_cast<size_t>(num_layers_) * 2 * num_kv_heads_ * head_dim_ * page_size_;
    return used * kv_size_per_page * sizeof(float);
}

size_t PagedKVCache::get_gpu_bytes() const {
    return gpu_bytes_;
}

int PagedKVCache::get_allocated_pages() const {
    return max_pages_;
}

int PagedKVCache::get_used_pages() const {
    int count = 0;
    for (const auto& page : pages_) {
        if (page.in_use)
            count++;
    }
    return count;
}

void PagedKVCache::print_status() const {
    std::cout << "[PagedKV] Status:" << std::endl;
    std::cout << "  Total pages:   " << max_pages_ << std::endl;
    std::cout << "  Used pages:    " << get_used_pages() << std::endl;
    std::cout << "  Active seqs:   " << seq_to_pages_.size() << std::endl;
    std::cout << "  KV bytes:      " << (get_kv_bytes() / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  GPU allocated: " << (gpu_bytes_ / (1024 * 1024)) << " MB" << std::endl;
}

} // namespace cuda
} // namespace qwen
