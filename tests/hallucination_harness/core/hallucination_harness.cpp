#include "hallucination_harness.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <regex>
#include <chrono>
#include <iostream>

namespace qwen {
namespace hallucination {

HallucinationHarness::HallucinationHarness(const HarnessConfig& config)
    : config(config), total_validations_(0), cumulative_score_(0.0f) {}

HallucinationHarness::~HallucinationHarness() = default;

void HallucinationHarness::register_validator(
    std::unique_ptr<IHallucinationValidator> validator) {
    validators_.push_back(std::move(validator));
}

ValidationReport HallucinationHarness::validate(const std::string& prompt,
                                               const std::string& response) {
    auto start = std::chrono::high_resolution_clock::now();

    ValidationReport report;
    report.prompt = prompt;
    report.response = response;
    report.total_hallucinations = 0;
    report.hallucination_score = 0.0f;
    report.passed = true;

    if (response.length() < config.min_response_length) {
        if (config.verbose) {
            std::cerr << "Response too short: " << response.length()
                      << " chars (min: " << config.min_response_length << ")\n";
        }
    }

    if (response.length() > config.max_response_length) {
        if (config.verbose) {
            std::cerr << "Response too long: " << response.length()
                      << " chars (max: " << config.max_response_length << ")\n";
        }
    }

    for (auto& validator : validators_) {
        auto result = validator->validate(prompt, response);
        if (result.is_hallucination) {
            report.hallucinations.push_back(result);
            report.total_hallucinations++;
            report.hallucination_score += result.confidence;
        }
        last_results_.push_back(result);
    }

    if (!report.hallucinations.empty()) {
        report.hallucination_score /= static_cast<float>(report.hallucinations.size());
        report.passed = report.hallucination_score < config.hallucination_threshold;
    }

    auto end = std::chrono::high_resolution_clock::now();
    report.validation_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    total_validations_++;
    cumulative_score_ += report.hallucination_score;

    return report;
}

std::vector<ValidationReport> HallucinationHarness::batch_validate(
    const std::vector<std::pair<std::string, std::string>>& inputs) {
    std::vector<ValidationReport> reports;
    reports.reserve(inputs.size());

    for (const auto& [prompt, response] : inputs) {
        reports.push_back(validate(prompt, response));
    }

    return reports;
}

float HallucinationHarness::get_overall_hallucination_score() const {
    if (total_validations_ == 0) return 0.0f;
    return cumulative_score_ / static_cast<float>(total_validations_);
}

int HallucinationHarness::get_total_validations() const {
    return total_validations_;
}

void HallucinationHarness::reset_stats() {
    total_validations_ = 0;
    cumulative_score_ = 0.0f;
    last_results_.clear();
}

FactualValidator::FactualValidator() {}

void FactualValidator::add_verified_fact(const std::string& fact) {
    std::string lower_fact = fact;
    std::transform(lower_fact.begin(), lower_fact.end(), lower_fact.begin(), ::tolower);
    verified_facts_.insert(lower_fact);
}

void FactualValidator::add_verified_facts(const std::vector<std::string>& facts) {
    for (const auto& fact : facts) {
        add_verified_fact(fact);
    }
}

bool FactualValidator::load_facts_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line[0] != '#') {
            add_verified_fact(line);
        }
    }

    return true;
}

std::vector<std::string> FactualValidator::extract_claims(const std::string& text) {
    std::vector<std::string> claims;
    std::regex claim_pattern(R"((?:is|are|was|were|has|have|had|can|could|will|would|should|may|might)\s+\w+)",
                            std::regex::icase);
    auto claims_begin = std::sregex_iterator(text.begin(), text.end(), claim_pattern);
    auto claims_end = std::sregex_iterator();

    for (auto it = claims_begin; it != claims_end; ++it) {
        claims.push_back(it->str());
    }

    return claims;
}

bool FactualValidator::contains_negation(const std::string& claim) {
    std::vector<std::string> negations = {"not", "no", "never", "n't", "cannot", "without"};
    std::string lower_claim = claim;
    std::transform(lower_claim.begin(), lower_claim.end(), lower_claim.begin(), ::tolower);

    for (const auto& neg : negations) {
        if (lower_claim.find(neg) != std::string::npos) {
            return true;
        }
    }
    return false;
}

HallucinationResult FactualValidator::validate(const std::string& prompt,
                                               const std::string& response) {
    HallucinationResult result;
    result.is_hallucination = false;
    result.type = HallucinationType::NONE;
    result.confidence = 0.0f;

    if (verified_facts_.empty()) {
        return result;
    }

    auto claims = extract_claims(response);

    for (const auto& claim : claims) {
        std::string lower_claim = claim;
        std::transform(lower_claim.begin(), lower_claim.end(), lower_claim.begin(), ::tolower);

        bool found_contradiction = false;
        for (const auto& verified : verified_facts_) {
            if (lower_claim.find(verified) != std::string::npos) {
                if (contains_negation(claim)) {
                    result.is_hallucination = true;
                    result.type = HallucinationType::FACTUAL_CONTRADICTION;
                    result.confidence = 0.9f;
                    result.description = "Negation of verified fact detected";
                    result.evidence = claim + " contradicts " + verified;
                    break;
                }
            }
        }

        if (found_contradiction) break;
    }

    return result;
}

NumericValidator::NumericValidator() : tolerance_(1e-6f) {}

std::vector<NumericValidator::NumericValue> NumericValidator::extract_numbers(
    const std::string& text) {
    std::vector<NumericValue> numbers;
    std::regex number_pattern(R"(-?\d+\.?\d*)");
    auto num_begin = std::sregex_iterator(text.begin(), text.end(), number_pattern);
    auto num_end = std::sregex_iterator();

    for (auto it = num_begin; it != num_end; ++it) {
        NumericValue nv;
        nv.raw_text = it->str();
        nv.value = std::stod(nv.raw_text);
        nv.position = it->position();
        numbers.push_back(nv);
    }

    return numbers;
}

bool NumericValidator::numbers_consistent(
    const std::vector<NumericValue>& prompt_nums,
    const std::vector<NumericValue>& response_nums) {
    if (prompt_nums.empty() || response_nums.empty()) {
        return true;
    }

    for (const auto& resp_num : response_nums) {
        for (const auto& prompt_num : prompt_nums) {
            if (std::abs(resp_num.value - prompt_num.value) < tolerance_) {
                return true;
            }
        }
    }

    return response_nums.size() <= 2;
}

HallucinationResult NumericValidator::validate(const std::string& prompt,
                                               const std::string& response) {
    HallucinationResult result;
    result.is_hallucination = false;
    result.type = HallucinationType::NONE;
    result.confidence = 0.0f;

    auto prompt_nums = extract_numbers(prompt);
    auto response_nums = extract_numbers(response);

    if (!numbers_consistent(prompt_nums, response_nums)) {
        result.is_hallucination = true;
        result.type = HallucinationType::NUMERIC_INCONSISTENCY;
        result.confidence = 0.85f;
        result.description = "Numeric values in response inconsistent with prompt";
        result.start_position = 0;
        result.end_position = static_cast<int>(response.length());
    }

    return result;
}

ConsistencyValidator::ConsistencyValidator() {}

std::vector<std::string> ConsistencyValidator::extract_entities(const std::string& text) {
    std::vector<std::string> entities;
    std::regex entity_pattern(R"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b)");
    auto entity_begin = std::sregex_iterator(text.begin(), text.end(), entity_pattern);
    auto entity_end = std::sregex_iterator();

    for (auto it = entity_begin; it != entity_end; ++it) {
        entities.push_back(it->str());
    }

    return entities;
}

std::unordered_map<std::string, std::string> ConsistencyValidator::extract_entity_attributes(
    const std::string& text) {
    std::unordered_map<std::string, std::string> attrs;

    auto entities = extract_entities(text);
    for (const auto& entity : entities) {
        std::regex attr_pattern(entity + R"(\s+is\s+(.+?)(?:\.|$))", std::regex::icase);
        auto attr_begin = std::sregex_iterator(text.begin(), text.end(), attr_pattern);
        auto attr_end = std::sregex_iterator();

        for (auto it = attr_begin; it != attr_end; ++it) {
            attrs[entity] = it->str(1);
        }
    }

    return attrs;
}

bool ConsistencyValidator::check_self_consistency(
    const std::unordered_map<std::string, std::string>& attrs) {
    std::unordered_map<std::string, std::vector<std::string>> entity_claims;

    for (const auto& [entity, claim] : attrs) {
        entity_claims[entity].push_back(claim);
    }

    for (const auto& [entity, claims] : entity_claims) {
        if (claims.size() > 1) {
            for (size_t i = 0; i < claims.size(); ++i) {
                for (size_t j = i + 1; j < claims.size(); ++j) {
                    std::string lower_i = claims[i];
                    std::string lower_j = claims[j];
                    std::transform(lower_i.begin(), lower_i.end(), lower_i.begin(), ::tolower);
                    std::transform(lower_j.begin(), lower_j.end(), lower_j.begin(), ::tolower);

                    if (lower_i != lower_j && lower_i.find(lower_j) == std::string::npos &&
                        lower_j.find(lower_i) == std::string::npos) {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

HallucinationResult ConsistencyValidator::validate(const std::string& prompt,
                                                   const std::string& response) {
    HallucinationResult result;
    result.is_hallucination = false;
    result.type = HallucinationType::NONE;
    result.confidence = 0.0f;

    auto prompt_attrs = extract_entity_attributes(prompt);
    auto response_attrs = extract_entity_attributes(response);

    std::unordered_map<std::string, std::string> all_attrs = prompt_attrs;
    for (const auto& [entity, attr] : response_attrs) {
        if (all_attrs.find(entity) != all_attrs.end()) {
            if (all_attrs[entity] != attr) {
                result.is_hallucination = true;
                result.type = HallucinationType::SELF_CONTRADICTION;
                result.confidence = 0.8f;
                result.description = "Entity has conflicting attributes";
                result.evidence = entity + " has both \"" + all_attrs[entity] +
                                  "\" and \"" + attr + "\"";
                break;
            }
        }
    }

    if (!result.is_hallucination) {
        auto combined_attrs = prompt_attrs;
        combined_attrs.insert(response_attrs.begin(), response_attrs.end());
        if (!check_self_consistency(combined_attrs)) {
            result.is_hallucination = true;
            result.type = HallucinationType::SELF_CONTRADICTION;
            result.confidence = 0.75f;
            result.description = "Self-contradiction detected in entity attributes";
        }
    }

    return result;
}

RepetitionValidator::RepetitionValidator() : threshold_(0.3f) {}

float RepetitionValidator::calculate_repetition_ratio(const std::string& text) {
    if (text.empty()) return 0.0f;

    auto phrases = find_repeated_phrases(text, 5);
    if (phrases.empty()) return 0.0f;

    int total_occurrences = 0;
    for (const auto& [phrase, count] : phrases) {
        total_occurrences += count;
    }

    return static_cast<float>(total_occurrences) / text.length();
}

std::vector<std::pair<std::string, int>> RepetitionValidator::find_repeated_phrases(
    const std::string& text, int min_length) {
    std::vector<std::pair<std::string, int>> repeated;
    std::vector<std::string> words;
    std::istringstream iss(text);
    std::string word;

    while (iss >> word) {
        words.push_back(word);
    }

    for (size_t len = min_length; len <= words.size() / 2; ++len) {
        std::unordered_map<std::string, int> phrase_count;
        for (size_t i = 0; i + len <= words.size(); ++i) {
            std::string phrase;
            for (size_t j = i; j < i + len; ++j) {
                phrase += words[j] + " ";
            }
            phrase.pop_back();
            phrase_count[phrase]++;
        }

        for (const auto& [phrase, count] : phrase_count) {
            if (count > 1) {
                repeated.push_back({phrase, count});
            }
        }
    }

    return repeated;
}

HallucinationResult RepetitionValidator::validate(const std::string& prompt,
                                                  const std::string& response) {
    HallucinationResult result;
    result.is_hallucination = false;
    result.type = HallucinationType::NONE;
    result.confidence = 0.0f;

    float ratio = calculate_repetition_ratio(response);

    if (ratio > threshold_) {
        result.is_hallucination = true;
        result.type = HallucinationType::REPETITION;
        result.confidence = std::min(ratio, 1.0f);
        result.description = "Excessive repetition detected in response";
        result.evidence = "Repetition ratio: " + std::to_string(ratio);

        auto repeated = find_repeated_phrases(response, 5);
        for (const auto& [phrase, count] : repeated) {
            if (count > 2) {
                result.related_tokens.push_back(phrase);
                break;
            }
        }
    }

    return result;
}

SemanticValidator::SemanticValidator() {}

std::vector<std::string> SemanticValidator::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::regex token_pattern(R"(\b\w+\b)");
    auto token_begin = std::sregex_iterator(text.begin(), text.end(), token_pattern);
    auto token_end = std::sregex_iterator();

    for (auto it = token_begin; it != token_end; ++it) {
        tokens.push_back(it->str());
    }

    return tokens;
}

float SemanticValidator::calculate_semantic_alignment(const std::string& prompt,
                                                      const std::string& response) {
    auto prompt_tokens = tokenize(prompt);
    auto response_tokens = tokenize(response);

    if (prompt_tokens.empty() || response_tokens.empty()) {
        return 0.0f;
    }

    std::unordered_set<std::string> prompt_set(prompt_tokens.begin(), prompt_tokens.end());
    int overlap = 0;

    for (const auto& token : response_tokens) {
        if (prompt_set.find(token) != prompt_set.end()) {
            overlap++;
        }
    }

    return static_cast<float>(overlap) / static_cast<float>(response_tokens.size());
}

bool SemanticValidator::is_response_related_to_prompt(const std::string& prompt,
                                                      const std::string& response) {
    float alignment = calculate_semantic_alignment(prompt, response);
    return alignment > 0.1f;
}

HallucinationResult SemanticValidator::validate(const std::string& prompt,
                                               const std::string& response) {
    HallucinationResult result;
    result.is_hallucination = false;
    result.type = HallucinationType::NONE;
    result.confidence = 0.0f;

    if (!is_response_related_to_prompt(prompt, response)) {
        result.is_hallucination = true;
        result.type = HallucinationType::UNSUPPORTED_INFERENCE;
        result.confidence = 0.7f;
        result.description = "Response appears unrelated to prompt";
        result.evidence = "Semantic alignment score too low";
    }

    return result;
}

EntityFabricationValidator::EntityFabricationValidator() {}

void EntityFabricationValidator::load_known_entities(const std::vector<std::string>& entities) {
    for (const auto& entity : entities) {
        std::string lower = entity;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        known_entities_.insert(lower);
    }
}

void EntityFabricationValidator::load_entities_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line[0] != '#') {
            std::string lower = line;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            known_entities_.insert(lower);
        }
    }
}

std::vector<std::string> EntityFabricationValidator::extract_proper_nouns(const std::string& text) {
    std::vector<std::string> proper_nouns;
    std::regex proper_pattern(R"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b)");
    auto noun_begin = std::sregex_iterator(text.begin(), text.end(), proper_pattern);
    auto noun_end = std::sregex_iterator();

    for (auto it = noun_begin; it != noun_end; ++it) {
        proper_nouns.push_back(it->str());
    }

    return proper_nouns;
}

bool EntityFabricationValidator::is_known_entity(const std::string& entity) {
    std::string lower = entity;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return known_entities_.find(lower) != known_entities_.end();
}

std::vector<std::string> EntityFabricationValidator::find_unknown_entities(
    const std::string& prompt, const std::string& response) {
    auto prompt_entities = extract_proper_nouns(prompt);
    auto response_entities = extract_proper_nouns(response);

    std::unordered_set<std::string> prompt_set(prompt_entities.begin(), prompt_entities.end());
    std::vector<std::string> unknown;

    for (const auto& entity : response_entities) {
        if (prompt_set.find(entity) == prompt_set.end() && !is_known_entity(entity)) {
            unknown.push_back(entity);
        }
    }

    return unknown;
}

HallucinationResult EntityFabricationValidator::validate(const std::string& prompt,
                                                         const std::string& response) {
    HallucinationResult result;
    result.is_hallucination = false;
    result.type = HallucinationType::NONE;
    result.confidence = 0.0f;

    if (known_entities_.empty()) {
        return result;
    }

    auto unknown_entities = find_unknown_entities(prompt, response);

    if (!unknown_entities.empty()) {
        float ratio = static_cast<float>(unknown_entities.size()) /
                      static_cast<float>(std::max(1, static_cast<int>(extract_proper_nouns(response).size())));

        if (ratio > 0.5f) {
            result.is_hallucination = true;
            result.type = HallucinationType::ENTITY_FABRICATION;
            result.confidence = std::min(ratio, 1.0f);
            result.description = "Many potentially fabricated entities detected";
            result.evidence = "Unknown entities: " + std::to_string(unknown_entities.size());
            result.related_tokens = unknown_entities;
        }
    }

    return result;
}

} // namespace hallucination
} // namespace qwen