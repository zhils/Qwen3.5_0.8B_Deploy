#include "hallucination_checkers.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <regex>
#include <iostream>

namespace qwen {
namespace hallucination {

TemporalConsistencyChecker::TemporalConsistencyChecker() {}

std::vector<std::pair<std::string, int>> TemporalConsistencyChecker::extract_dated_entities(
    const std::string& text) {
    std::vector<std::pair<std::string, int>> dated_entities;

    std::regex pattern(R"(\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+\(?(1[0-9]{3}|20[0-2][0-9])\)?)");
    auto begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        dated_entities.push_back({it->str(1), std::stoi(it->str(2))});
    }

    return dated_entities;
}

bool TemporalConsistencyChecker::check_temporal_consistency(const std::string& entity,
                                                            int year,
                                                            const std::string& claim) {
    auto it = temporal_facts_.find(entity);
    if (it == temporal_facts_.end()) {
        return true;
    }

    for (const auto& fact : it->second) {
        if (fact.year > year) {
            return false;
        }
    }

    return true;
}

HallucinationResult TemporalConsistencyChecker::check(const std::string& prompt,
                                                      const std::string& response) {
    HallucinationResult result;
    result.is_hallucination = false;
    result.type = HallucinationType::TEMPORAL_INCONSISTENCY;
    result.confidence = 0.0f;

    auto dated = extract_dated_entities(response);

    for (const auto& [entity, year] : dated) {
        auto entity_claims = extract_dated_entities(prompt);
        bool found_in_prompt = false;

        for (const auto& [prompt_entity, prompt_year] : entity_claims) {
            if (prompt_entity == entity && prompt_year != year) {
                result.is_hallucination = true;
                result.type = HallucinationType::TEMPORAL_INCONSISTENCY;
                result.confidence = 0.85f;
                result.description = "Temporal inconsistency detected for entity: " + entity;
                result.evidence = "Prompt states " + std::to_string(prompt_year) +
                                  " but response states " + std::to_string(year);
                break;
            }
            if (prompt_entity == entity) {
                found_in_prompt = true;
            }
        }

        if (!found_in_prompt && !temporal_facts_.empty()) {
            if (!check_temporal_consistency(entity, year, response)) {
                result.is_hallucination = true;
                result.confidence = 0.8f;
                result.description = "Temporal fact violated for: " + entity;
            }
        }

        if (result.is_hallucination) break;
    }

    return result;
}

void TemporalConsistencyChecker::add_temporal_fact(const std::string& entity, int year,
                                                  const std::string& fact) {
    temporal_facts_[entity].push_back({year, fact});
}

AttributionChecker::AttributionChecker() {}

std::vector<std::string> AttributionChecker::extract_citations(const std::string& text) {
    std::vector<std::string> citations;

    std::regex pattern(R"lit("([^"]+)")lit");
    auto begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        citations.push_back(it->str(1));
    }

    return citations;
}

std::vector<std::string> AttributionChecker::extract_quoted_statements(const std::string& text) {
    return extract_citations(text);
}

bool AttributionChecker::is_claim_supported(const std::string& claim, const std::string& source) {
    std::string lower_claim = claim;
    std::string lower_source = source;
    std::transform(lower_claim.begin(), lower_claim.end(), lower_claim.begin(), ::tolower);
    std::transform(lower_source.begin(), lower_source.end(), lower_source.begin(), ::tolower);

    return lower_source.find(lower_claim) != std::string::npos;
}

HallucinationResult AttributionChecker::check(const std::string& prompt,
                                             const std::string& response) {
    HallucinationResult result;
    result.is_hallucination = false;
    result.type = HallucinationType::UNSUPPORTED_INFERENCE;
    result.confidence = 0.0f;

    if (sources_.empty()) {
        return result;
    }

    auto quoted = extract_quoted_statements(response);

    for (const auto& quote : quoted) {
        bool supported = false;

        for (const auto& [name, source] : sources_) {
            if (is_claim_supported(quote, source.name)) {
                supported = true;
                break;
            }
        }

        if (!supported) {
            result.is_hallucination = true;
            result.confidence = 0.75f;
            result.description = "Quote/attribution not supported by known sources";
            result.evidence = "Unsupported quote: \"" + quote + "\"";
            result.related_tokens.push_back(quote);
            break;
        }
    }

    return result;
}

void AttributionChecker::add_source(const std::string& source_name,
                                    const std::vector<std::string>& claims) {
    Source source;
    source.name = source_name;
    for (const auto& claim : claims) {
        std::string lower = claim;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        source.claims.insert(lower);
    }
    sources_[source_name] = source;
}

void AttributionChecker::load_sources_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return;
    }

    std::string line;
    std::string current_source;
    std::vector<std::string> current_claims;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        if (line[0] == '#') {
            if (!current_source.empty()) {
                add_source(current_source, current_claims);
                current_claims.clear();
            }
            current_source = line.substr(1);
            current_source.erase(0, current_source.find_first_not_of(" \t\n\r"));
            current_source.erase(current_source.find_last_not_of(" \t\n\r") + 1);
        } else {
            current_claims.push_back(line);
        }
    }

    if (!current_source.empty()) {
        add_source(current_source, current_claims);
    }
}

LogicConsistencyChecker::LogicConsistencyChecker() {}

std::vector<LogicConsistencyChecker::LogicalStatement>
LogicConsistencyChecker::extract_logical_statements(const std::string& text) {
    std::vector<LogicalStatement> statements;

    std::regex pattern(R"(\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(is|are|was|were|can|could|will|would|should|may|might)\s+(not\s+)?(.+?)(?:\.|$))",
                       std::regex::icase);
    auto begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        LogicalStatement stmt;
        stmt.subject = it->str(1);
        stmt.predicate = it->str(2);
        stmt.is_negative = it->str(3).find("not") != std::string::npos;
        statements.push_back(stmt);
    }

    return statements;
}

bool LogicConsistencyChecker::statements_contradict(const LogicalStatement& a,
                                                   const LogicalStatement& b) {
    if (a.subject != b.subject || a.predicate != b.predicate) {
        return false;
    }

    return a.is_negative != b.is_negative;
}

bool LogicConsistencyChecker::check_logical_consistency(
    const std::vector<LogicalStatement>& statements) {
    for (size_t i = 0; i < statements.size(); ++i) {
        for (size_t j = i + 1; j < statements.size(); ++j) {
            if (statements_contradict(statements[i], statements[j])) {
                return false;
            }
        }
    }
    return true;
}

HallucinationResult LogicConsistencyChecker::check(const std::string& prompt,
                                                  const std::string& response) {
    HallucinationResult result;
    result.is_hallucination = false;
    result.type = HallucinationType::SELF_CONTRADICTION;
    result.confidence = 0.0f;

    auto prompt_stmts = extract_logical_statements(prompt);
    auto response_stmts = extract_logical_statements(response);

    for (const auto& resp_stmt : response_stmts) {
        for (const auto& prompt_stmt : prompt_stmts) {
            if (resp_stmt.subject == prompt_stmt.subject &&
                resp_stmt.predicate == prompt_stmt.predicate &&
                resp_stmt.is_negative != prompt_stmt.is_negative) {
                result.is_hallucination = true;
                result.confidence = 0.9f;
                result.description = "Logical contradiction between prompt and response";
                result.evidence = "Prompt says \"" + prompt_stmt.subject + " " +
                                  prompt_stmt.predicate + "\" but response says opposite";
                return result;
            }
        }
    }

    auto all_stmts = prompt_stmts;
    all_stmts.insert(all_stmts.end(), response_stmts.begin(), response_stmts.end());

    if (!check_logical_consistency(all_stmts)) {
        result.is_hallucination = true;
        result.confidence = 0.85f;
        result.description = "Self-contradiction detected in combined statements";
    }

    return result;
}

MathConsistencyChecker::MathConsistencyChecker() : tolerance_(1e-6) {}

std::vector<MathConsistencyChecker::MathExpression>
MathConsistencyChecker::extract_math_expressions(const std::string& text) {
    std::vector<MathExpression> expressions;

    std::regex pattern(R"((\d+\.?\d*)\s*([+\-*/=])\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*))");
    auto begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        MathExpression expr;
        expr.expression = it->str(0);
        expr.result = std::stod(it->str(4));
        expr.position = it->position();

        double a = std::stod(it->str(1));
        double b = std::stod(it->str(3));
        char op = it->str(2)[0];

        double calculated = 0;
        switch (op) {
            case '+': calculated = a + b; break;
            case '-': calculated = a - b; break;
            case '*': calculated = a * b; break;
            case '/': calculated = (b != 0) ? a / b : 0; break;
        }

        if (std::abs(calculated - expr.result) > tolerance_) {
            expressions.push_back(expr);
        }
    }

    return expressions;
}

std::optional<double> MathConsistencyChecker::evaluate_simple_expression(
    const std::string& expr) {
    std::istringstream iss(expr);
    double a, b;
    char op;

    if (!(iss >> a >> op >> b)) {
        return std::nullopt;
    }

    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/': return (b != 0) ? a / b : std::nullopt;
        default: return std::nullopt;
    }
}

bool MathConsistencyChecker::check_math_consistency(const std::vector<MathExpression>& exprs) {
    return exprs.empty();
}

HallucinationResult MathConsistencyChecker::check(const std::string& prompt,
                                                const std::string& response) {
    HallucinationResult result;
    result.is_hallucination = false;
    result.type = HallucinationType::NUMERIC_INCONSISTENCY;
    result.confidence = 0.0f;

    auto prompt_exprs = extract_math_expressions(prompt);
    auto response_exprs = extract_math_expressions(response);

    for (const auto& expr : response_exprs) {
        result.is_hallucination = true;
        result.confidence = 0.95f;
        result.description = "Mathematical error detected";
        result.evidence = "Expression \"" + expr.expression +
                          "\" claims result is " + std::to_string(expr.result);
        result.start_position = expr.position;
        result.end_position = expr.position + static_cast<int>(expr.expression.length());
        break;
    }

    return result;
}

SubjectDriftChecker::SubjectDriftChecker() : threshold_(0.3f) {}

std::vector<std::string> SubjectDriftChecker::extract_topics(const std::string& text) {
    std::vector<std::string> topics;
    std::regex pattern(R"(\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b)");
    auto begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        std::string topic = it->str(1);
        if (topic.length() > 3) {
            topics.push_back(topic);
        }
    }

    return topics;
}

float SubjectDriftChecker::calculate_topic_overlap(
    const std::vector<std::string>& prompt_topics,
    const std::vector<std::string>& response_topics) {
    if (prompt_topics.empty() || response_topics.empty()) {
        return 1.0f;
    }

    std::unordered_set<std::string> prompt_set(prompt_topics.begin(), prompt_topics.end());
    int overlap = 0;

    for (const auto& topic : response_topics) {
        if (prompt_set.find(topic) != prompt_set.end()) {
            overlap++;
        }
    }

    return static_cast<float>(overlap) / static_cast<float>(response_topics.size());
}

int SubjectDriftChecker::find_drift_point(const std::vector<std::string>& prompt_topics,
                                         const std::vector<std::string>& response_topics) {
    std::unordered_set<std::string> prompt_set(prompt_topics.begin(), prompt_topics.end());

    for (size_t i = 0; i < response_topics.size(); ++i) {
        if (prompt_set.find(response_topics[i]) == prompt_set.end()) {
            return static_cast<int>(i);
        }
    }

    return -1;
}

HallucinationResult SubjectDriftChecker::check(const std::string& prompt,
                                             const std::string& response) {
    HallucinationResult result;
    result.is_hallucination = false;
    result.type = HallucinationType::UNSUPPORTED_INFERENCE;
    result.confidence = 0.0f;

    auto prompt_topics = extract_topics(prompt);
    auto response_topics = extract_topics(response);

    if (prompt_topics.empty() || response_topics.empty()) {
        return result;
    }

    float overlap = calculate_topic_overlap(prompt_topics, response_topics);

    if (overlap < threshold_) {
        result.is_hallucination = true;
        result.type = HallucinationType::UNSUPPORTED_INFERENCE;
        result.confidence = 1.0f - overlap;
        result.description = "Significant topic drift detected";
        result.evidence = "Topic overlap only " + std::to_string(static_cast<int>(overlap * 100)) + "%";

        int drift_point = find_drift_point(prompt_topics, response_topics);
        if (drift_point >= 0 && static_cast<size_t>(drift_point) < response_topics.size()) {
            result.related_tokens.push_back(response_topics[drift_point]);
        }
    }

    return result;
}

} // namespace hallucination
} // namespace qwen