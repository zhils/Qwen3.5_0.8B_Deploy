#ifndef HALLUCINATION_HARNESS_HPP
#define HALLUCINATION_HARNESS_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <functional>
#include <optional>
#include <chrono>

namespace qwen {
namespace hallucination {

enum class HallucinationType {
    NONE = 0,
    FACTUAL_CONTRADICTION = 1,
    NUMERIC_INCONSISTENCY = 2,
    ENTITY_FABRICATION = 3,
    SELF_CONTRADICTION = 4,
    REPETITION = 5,
    SEMANTIC_INCONSISTENCY = 6,
    TEMPORAL_INCONSISTENCY = 7,
    UNSUPPORTED_INFERENCE = 8
};

struct HallucinationResult {
    bool is_hallucination;
    HallucinationType type;
    float confidence;
    std::string description;
    std::string evidence;
    std::vector<std::string> related_tokens;
    int start_position;
    int end_position;
};

struct ValidationReport {
    std::string prompt;
    std::string response;
    std::vector<HallucinationResult> hallucinations;
    int total_hallucinations;
    float hallucination_score;
    bool passed;
    std::chrono::milliseconds validation_time;
};

struct HarnessConfig {
    float hallucination_threshold = 0.7f;
    bool enable_factual_check = true;
    bool enable_numeric_check = true;
    bool enable_consistency_check = true;
    bool enable_repetition_check = true;
    bool enable_semantic_check = true;
    int max_response_length = 4096;
    int min_response_length = 4;
    float repetition_threshold = 0.3f;
    float numeric_tolerance = 1e-6f;
    bool verbose = false;
};

class IHallucinationValidator {
public:
    virtual ~IHallucinationValidator() = default;
    virtual HallucinationResult validate(const std::string& prompt,
                                         const std::string& response) = 0;
    virtual std::string name() const = 0;
};

class HallucinationHarness {
public:
    explicit HallucinationHarness(const HarnessConfig& config = HarnessConfig());
    ~HallucinationHarness();

    void register_validator(std::unique_ptr<IHallucinationValidator> validator);
    ValidationReport validate(const std::string& prompt, const std::string& response);
    std::vector<ValidationReport> batch_validate(
        const std::vector<std::pair<std::string, std::string>>& inputs);

    float get_overall_hallucination_score() const;
    int get_total_validations() const;
    void reset_stats();

    HarnessConfig config;
    std::vector<std::unique_ptr<IHallucinationValidator>> validators_;

private:
    int total_validations_;
    float cumulative_score_;
    std::vector<HallucinationResult> last_results_;
};

class FactualValidator : public IHallucinationValidator {
public:
    FactualValidator();
    HallucinationResult validate(const std::string& prompt,
                                  const std::string& response) override;
    std::string name() const override { return "FactualValidator"; }

    void add_verified_fact(const std::string& fact);
    void add_verified_facts(const std::vector<std::string>& facts);
    bool load_facts_from_file(const std::string& filepath);

private:
    std::unordered_set<std::string> verified_facts_;
    std::vector<std::string> extract_claims(const std::string& text);
    bool contains_negation(const std::string& claim);
};

class NumericValidator : public IHallucinationValidator {
public:
    NumericValidator();
    HallucinationResult validate(const std::string& prompt,
                                  const std::string& response) override;
    std::string name() const override { return "NumericValidator"; }

    void set_tolerance(float tol) { tolerance_ = tol; }

private:
    struct NumericValue {
        std::string raw_text;
        double value;
        int position;
    };

    std::vector<NumericValue> extract_numbers(const std::string& text);
    bool numbers_consistent(const std::vector<NumericValue>& prompt_nums,
                            const std::vector<NumericValue>& response_nums);
    float tolerance_;
};

class ConsistencyValidator : public IHallucinationValidator {
public:
    ConsistencyValidator();
    HallucinationResult validate(const std::string& prompt,
                                  const std::string& response) override;
    std::string name() const override { return "ConsistencyValidator"; }

private:
    std::unordered_map<std::string, std::string> extract_entity_attributes(
        const std::string& text);
    bool check_self_consistency(
        const std::unordered_map<std::string, std::string>& attrs);
    std::vector<std::string> extract_entities(const std::string& text);
};

class RepetitionValidator : public IHallucinationValidator {
public:
    RepetitionValidator();
    HallucinationResult validate(const std::string& prompt,
                                  const std::string& response) override;
    std::string name() const override { return "RepetitionValidator"; }

    void set_threshold(float th) { threshold_ = th; }

private:
    float calculate_repetition_ratio(const std::string& text);
    std::vector<std::pair<std::string, int>> find_repeated_phrases(
        const std::string& text, int min_length = 5);
    float threshold_;
};

class SemanticValidator : public IHallucinationValidator {
public:
    SemanticValidator();
    HallucinationResult validate(const std::string& prompt,
                                  const std::string& response) override;
    std::string name() const override { return "SemanticValidator"; }

private:
    std::vector<std::string> tokenize(const std::string& text);
    float calculate_semantic_alignment(const std::string& prompt,
                                       const std::string& response);
    bool is_response_related_to_prompt(const std::string& prompt,
                                        const std::string& response);
};

class EntityFabricationValidator : public IHallucinationValidator {
public:
    EntityFabricationValidator();
    HallucinationResult validate(const std::string& prompt,
                                  const std::string& response) override;
    std::string name() const override { return "EntityFabricationValidator"; }

    void load_known_entities(const std::vector<std::string>& entities);
    void load_entities_from_file(const std::string& filepath);

private:
    std::vector<std::string> extract_proper_nouns(const std::string& text);
    bool is_known_entity(const std::string& entity);
    std::vector<std::string> find_unknown_entities(
        const std::string& prompt, const std::string& response);
    std::unordered_set<std::string> known_entities_;
};

} // namespace hallucination
} // namespace qwen

#endif // HALLUCINATION_HARNESS_HPP