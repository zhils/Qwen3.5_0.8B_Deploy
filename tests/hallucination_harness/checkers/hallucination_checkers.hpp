#ifndef HALLUCINATION_CHECKERS_HPP
#define HALLUCINATION_CHECKERS_HPP

#include "hallucination_harness.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace qwen {
namespace hallucination {

class TemporalConsistencyChecker {
public:
    TemporalConsistencyChecker();

    HallucinationResult check(const std::string& prompt, const std::string& response);

    void add_temporal_fact(const std::string& entity, int year, const std::string& fact);
    void clear() { temporal_facts_.clear(); }

private:
    struct TemporalFact {
        int year;
        std::string fact;
    };

    std::unordered_map<std::string, std::vector<TemporalFact>> temporal_facts_;

    std::vector<std::pair<std::string, int>> extract_dated_entities(const std::string& text);
    bool check_temporal_consistency(const std::string& entity, int year, const std::string& claim);
};

class AttributionChecker {
public:
    AttributionChecker();

    HallucinationResult check(const std::string& prompt, const std::string& response);

    void add_source(const std::string& source_name, const std::vector<std::string>& claims);
    void load_sources_from_file(const std::string& filepath);
    void clear_sources() { sources_.clear(); }

private:
    struct Source {
        std::string name;
        std::unordered_set<std::string> claims;
    };

    std::unordered_map<std::string, Source> sources_;

    std::vector<std::string> extract_citations(const std::string& text);
    bool is_claim_supported(const std::string& claim, const std::string& source);
    std::vector<std::string> extract_quoted_statements(const std::string& text);
};

class LogicConsistencyChecker {
public:
    LogicConsistencyChecker();

    HallucinationResult check(const std::string& prompt, const std::string& response);

private:
    struct LogicalStatement {
        std::string subject;
        std::string predicate;
        bool is_negative;
    };

    std::vector<LogicalStatement> extract_logical_statements(const std::string& text);
    bool check_logical_consistency(const std::vector<LogicalStatement>& statements);
    bool statements_contradict(const LogicalStatement& a, const LogicalStatement& b);
};

class MathConsistencyChecker {
public:
    MathConsistencyChecker();

    HallucinationResult check(const std::string& prompt, const std::string& response);

    void set_tolerance(double tol) { tolerance_ = tol; }

private:
    struct MathExpression {
        std::string expression;
        double result;
        int position;
    };

    std::vector<MathExpression> extract_math_expressions(const std::string& text);
    std::optional<double> evaluate_simple_expression(const std::string& expr);
    bool check_math_consistency(const std::vector<MathExpression>& exprs);

    double tolerance_;
};

class SubjectDriftChecker {
public:
    SubjectDriftChecker();

    HallucinationResult check(const std::string& prompt, const std::string& response);

    void set_threshold(float threshold) { threshold_ = threshold; }

private:
    float threshold_;

    std::vector<std::string> extract_topics(const std::string& text);
    float calculate_topic_overlap(const std::vector<std::string>& prompt_topics,
                                  const std::vector<std::string>& response_topics);
    int find_drift_point(const std::vector<std::string>& prompt_topics,
                        const std::vector<std::string>& response_topics);
};

} // namespace hallucination
} // namespace qwen

#endif // HALLUCINATION_CHECKERS_HPP