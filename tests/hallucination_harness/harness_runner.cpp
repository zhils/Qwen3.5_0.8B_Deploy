#include "hallucination_harness.hpp"
#include "hallucination_checkers.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

namespace qwen {
namespace hallucination {

class HallucinationHarnessRunner {
public:
    HallucinationHarnessRunner(const HarnessConfig& config = HarnessConfig())
        : harness_(config) {
        setup_validators();
    }

    void setup_validators() {
        harness_.register_validator(std::make_unique<FactualValidator>());
        harness_.register_validator(std::make_unique<NumericValidator>());
        harness_.register_validator(std::make_unique<ConsistencyValidator>());
        harness_.register_validator(std::make_unique<RepetitionValidator>());
        harness_.register_validator(std::make_unique<SemanticValidator>());
        harness_.register_validator(std::make_unique<EntityFabricationValidator>());
    }

    void load_knowledge_base(const std::string& facts_file) {
        auto factual = std::make_unique<FactualValidator>();
        factual->load_facts_from_file(facts_file);
        harness_.register_validator(std::move(factual));
    }

    ValidationReport run_single_test(const std::string& prompt,
                                     const std::string& response) {
        return harness_.validate(prompt, response);
    }

    std::vector<ValidationReport> run_batch_tests(
        const std::vector<std::pair<std::string, std::string>>& tests) {
        return harness_.batch_validate(tests);
    }

    void print_report(const ValidationReport& report) {
        std::cout << "\n========================================\n";
        std::cout << "HALLUCINATION VALIDATION REPORT\n";
        std::cout << "========================================\n\n";

        std::cout << "Prompt: " << report.prompt.substr(0, 100)
                  << (report.prompt.length() > 100 ? "..." : "") << "\n\n";

        std::cout << "Response: " << report.response.substr(0, 100)
                  << (report.response.length() > 100 ? "..." : "") << "\n\n";

        std::cout << std::boolalpha;
        std::cout << "Passed: " << report.passed << "\n";
        std::cout << "Total Hallucinations: " << report.total_hallucinations << "\n";
        std::cout << "Hallucination Score: " << std::fixed << std::setprecision(3)
                  << report.hallucination_score << "\n";
        std::cout << "Validation Time: " << report.validation_time.count() << " ms\n\n";

        if (!report.hallucinations.empty()) {
            std::cout << "Detected Issues:\n";
            std::cout << "----------------------------------------\n";

            for (size_t i = 0; i < report.hallucinations.size(); ++i) {
                const auto& h = report.hallucinations[i];
                std::cout << "  [" << (i + 1) << "] " << get_type_name(h.type) << "\n";
                std::cout << "      Confidence: " << std::fixed << std::setprecision(2)
                          << (h.confidence * 100) << "%\n";
                std::cout << "      Description: " << h.description << "\n";
                if (!h.evidence.empty()) {
                    std::cout << "      Evidence: " << h.evidence << "\n";
                }
                std::cout << "\n";
            }
        }
    }

    void print_summary(const std::vector<ValidationReport>& reports) {
        int total_tests = static_cast<int>(reports.size());
        int passed = 0;
        float total_score = 0.0f;
        int total_hallucinations = 0;

        std::unordered_map<HallucinationType, int> type_counts;

        for (const auto& report : reports) {
            if (report.passed) passed++;
            total_score += report.hallucination_score;
            total_hallucinations += report.total_hallucinations;

            for (const auto& h : report.hallucinations) {
                type_counts[h.type]++;
            }
        }

        std::cout << "\n========================================\n";
        std::cout << "BATCH TEST SUMMARY\n";
        std::cout << "========================================\n\n";

        std::cout << "Total Tests: " << total_tests << "\n";
        std::cout << "Passed: " << passed << " (" << std::fixed << std::setprecision(1)
                  << (passed * 100.0f / total_tests) << "%)\n";
        std::cout << "Failed: " << (total_tests - passed) << "\n\n";

        std::cout << "Average Hallucination Score: "
                  << std::fixed << std::setprecision(3)
                  << (total_tests > 0 ? total_score / total_tests : 0) << "\n";
        std::cout << "Total Hallucinations Detected: " << total_hallucinations << "\n\n";

        if (!type_counts.empty()) {
            std::cout << "Hallucination Type Breakdown:\n";
            std::cout << "----------------------------------------\n";
            for (const auto& [type, count] : type_counts) {
                std::cout << "  " << get_type_name(type) << ": " << count << "\n";
            }
        }

        std::cout << "\n";
    }

    void export_json_report(const std::vector<ValidationReport>& reports,
                          const std::string& filepath) {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for JSON export: " << filepath << "\n";
            return;
        }

        file << "{\n";
        file << "  \"total_tests\": " << reports.size() << ",\n";
        file << "  \"passed\": " << std::count_if(reports.begin(), reports.end(),
            [](const auto& r) { return r.passed; }) << ",\n";

        float total_score = 0.0f;
        for (const auto& r : reports) total_score += r.hallucination_score;
        file << "  \"average_score\": " << std::fixed << std::setprecision(4)
             << (reports.empty() ? 0 : total_score / reports.size()) << ",\n";

        file << "  \"results\": [\n";
        for (size_t i = 0; i < reports.size(); ++i) {
            const auto& r = reports[i];
            file << "    {\n";
            file << "      \"prompt\": \"" << escape_json(r.prompt) << "\",\n";
            file << "      \"response\": \"" << escape_json(r.response) << "\",\n";
            file << "      \"passed\": " << std::boolalpha << r.passed << ",\n";
            file << "      \"score\": " << std::fixed << std::setprecision(4) << r.hallucination_score << ",\n";
            file << "      \"hallucination_count\": " << r.total_hallucinations << ",\n";
            file << "      \"validation_time_ms\": " << r.validation_time.count() << "\n";
            file << "    }" << (i < reports.size() - 1 ? "," : "") << "\n";
        }
        file << "  ]\n";
        file << "}\n";

        file.close();
        std::cout << "JSON report exported to: " << filepath << "\n";
    }

private:
    HallucinationHarness harness_;

    std::string get_type_name(HallucinationType type) {
        switch (type) {
            case HallucinationType::NONE: return "None";
            case HallucinationType::FACTUAL_CONTRADICTION: return "Factual Contradiction";
            case HallucinationType::NUMERIC_INCONSISTENCY: return "Numeric Inconsistency";
            case HallucinationType::ENTITY_FABRICATION: return "Entity Fabrication";
            case HallucinationType::SELF_CONTRADICTION: return "Self-Contradiction";
            case HallucinationType::REPETITION: return "Repetition";
            case HallucinationType::SEMANTIC_INCONSISTENCY: return "Semantic Inconsistency";
            case HallucinationType::TEMPORAL_INCONSISTENCY: return "Temporal Inconsistency";
            case HallucinationType::UNSUPPORTED_INFERENCE: return "Unsupported Inference";
            default: return "Unknown";
        }
    }

    std::string escape_json(const std::string& s) {
        std::string result;
        for (char c : s) {
            switch (c) {
                case '"': result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                default: result += c; break;
            }
        }
        return result;
    }
};

} // namespace hallucination
} // namespace qwen

using namespace qwen::hallucination;

int main(int argc, char** argv) {
    HarnessConfig config;
    config.hallucination_threshold = 0.7f;
    config.enable_repetition_check = true;
    config.enable_numeric_check = true;
    config.verbose = true;

    HallucinationHarnessRunner runner(config);

    std::vector<std::pair<std::string, std::string>> test_cases = {
        {
            "The capital of France is Paris. What is the capital of France?",
            "The capital of France is Paris. It is located in Western Europe."
        },
        {
            "What is 2 + 2?",
            "2 + 2 = 5. This is a common mathematical truth."
        },
        {
            "Tell me about dogs.",
            "Dogs are feline animals that bark. They have wings and can fly."
        },
        {
            "What is the capital of Japan?",
            "The capital of Japan is Tokyo. Tokyo is also the capital of China."
        },
        {
            "Tell me a short story.",
            "Once upon a time there was a cat. The cat was a dog. The cat barked. The cat was a cat. The cat was a cat. The cat was a cat."
        },
        {
            "What year did World War II end?",
            "World War II ended in 1945. The war began in 1939."
        },
        {
            "Who is the president of the United States?",
            "The current president is Joe Biden. The previous president was also Joe Biden."
        }
    };

    std::cout << "========================================\n";
    std::cout << "HALLUCINATION DETECTION HARNESS TEST\n";
    std::cout << "========================================\n";
    std::cout << "Running " << test_cases.size() << " test cases...\n";

    auto reports = runner.run_batch_tests(test_cases);

    for (size_t i = 0; i < reports.size(); ++i) {
        std::cout << "\n\n[Test Case " << (i + 1) << "/" << test_cases.size() << "]\n";
        runner.print_report(reports[i]);
    }

    runner.print_summary(reports);

    std::string output_path = "./hallucination_report.json";
    runner.export_json_report(reports, output_path);

    return 0;
}