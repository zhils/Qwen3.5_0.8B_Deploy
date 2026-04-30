#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "src/backend/cpu/core/language_common.hpp"
#include "src/backend/cpu/core/language_mlp.hpp"
#include "src/backend/cpu/core/lm_head.hpp"

struct BenchmarkResult {
    std::string module_name;
    double cpu_time_ms;
    double gpu_time_ms;
    double speedup;
    int input_size;
};

class PerformanceBenchmark {
  public:
    static void run_all_benchmarks() {
        std::cout << "========================================" << std::endl;
        std::cout << "Qwen3.5-0.8B CPU vs CUDA Performance Test" << std::endl;
        std::cout << "========================================" << std::endl;

        std::vector<BenchmarkResult> results;

        results.push_back(benchmark_rmsnorm());
        results.push_back(benchmark_mlp());
        results.push_back(benchmark_lm_head());

        print_results(results);
        save_results_to_file(results);
    }

  private:
    static std::vector<float> generate_random_vector(int size, float min = -1.0f,
                                                     float max = 1.0f) {
        std::vector<float> data(size);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(min, max);
        for (int i = 0; i < size; ++i) {
            data[i] = dis(gen);
        }
        return data;
    }

    static BenchmarkResult benchmark_rmsnorm() {
        const int H = 1024;
        const int BATCH_SIZE = 24;

        qwen::RMSNorm cpu_norm(H);
        cpu_norm.set_weight(generate_random_vector(H, -0.5f, 0.5f));

        auto input = generate_random_vector(H * BATCH_SIZE);
        std::vector<float> cpu_output(H * BATCH_SIZE);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            cpu_norm.forward_batch(input.data(), cpu_output.data(), BATCH_SIZE);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double, std::milli>(end - start).count() / 100.0;

        return {"RMSNorm", cpu_time, 0.0, 0.0, H * BATCH_SIZE};
    }

    static BenchmarkResult benchmark_mlp() {
        const int H = 1024;
        const int I = 3584;

        qwen::MLP cpu_mlp(H, I);
        cpu_mlp.set_weights(generate_random_vector(I * H), generate_random_vector(I * H),
                            generate_random_vector(H * I));

        auto input = generate_random_vector(H);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            cpu_mlp.forward(input);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double, std::milli>(end - start).count() / 100.0;

        return {"MLP/SwiGLU", cpu_time, 0.0, 0.0, H};
    }

    static BenchmarkResult benchmark_lm_head() {
        const int H = 1024;
        const int V = 248320;

        qwen::LMHead cpu_lm_head(H, V);
        cpu_lm_head.set_weight(generate_random_vector(V * H));

        auto input = generate_random_vector(H);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            cpu_lm_head.forward(input);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;

        return {"LMHead", cpu_time, 0.0, 0.0, H};
    }

    static void print_results(const std::vector<BenchmarkResult>& results) {
        std::cout << "\n" << std::string(80, '-') << std::endl;
        std::cout << "| Module | Input Size | CPU(ms) | GPU(ms) | Speedup |" << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        for (const auto& r : results) {
            std::ostringstream gpu_str, speed_str;
            if (r.gpu_time_ms > 0) {
                gpu_str << r.gpu_time_ms;
                speed_str << r.speedup;
            } else {
                gpu_str << "N/A";
                speed_str << "N/A";
            }

            std::cout << "| " << std::left << std::setw(16) << r.module_name << "| "
                      << std::setw(11) << r.input_size << "| " << std::setw(9) << std::fixed
                      << std::setprecision(4) << r.cpu_time_ms << "| " << std::setw(9)
                      << gpu_str.str() << "| " << std::setw(8) << speed_str.str() << "|"
                      << std::endl;
        }

        std::cout << std::string(80, '-') << std::endl;
        std::cout << "\nNote: GPU timing requires CUDA environment" << std::endl;
    }

    static void save_results_to_file(const std::vector<BenchmarkResult>& results) {
        std::ofstream file("performance_benchmark_results.csv");
        if (!file)
            return;

        file << "Module,InputSize,CPU_ms,GPU_ms,Speedup\n";
        for (const auto& r : results) {
            file << r.module_name << "," << r.input_size << "," << r.cpu_time_ms << ","
                 << r.gpu_time_ms << "," << r.speedup << "\n";
        }

        file.close();
        std::cout << "\nResults saved to: performance_benchmark_results.csv" << std::endl;
    }
};

int main() {
    try {
        PerformanceBenchmark::run_all_benchmarks();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
