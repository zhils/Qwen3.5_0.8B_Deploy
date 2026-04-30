# Hallucination Detection Harness

A comprehensive C++ harness for detecting and measuring hallucinations in LLM inference outputs.

## Architecture

```
tests/hallucination_harness/
├── core/
│   ├── hallucination_harness.hpp    # Core interfaces and classes
│   └── hallucination_harness.cpp     # Implementation
├── checkers/
│   ├── hallucination_checkers.hpp    # Advanced checker interfaces
│   └── hallucination_checkers.cpp     # Advanced checker implementations
├── data/
│   ├── knowledge_base.txt            # Verified facts for validation
│   └── test_cases.json               # Test case definitions
├── harness_runner.cpp                # Main test runner
└── README.md                        # This file
```

## Features

### Hallucination Types Detected

| Type | Description | Validator |
|------|-------------|-----------|
| `FACTUAL_CONTRADICTION` | Response contradicts verified facts | `FactualValidator` |
| `NUMERIC_INCONSISTENCY` | Numerical values inconsistent | `NumericValidator` |
| `ENTITY_FABRICATION` | Fabricated entities not in prompt | `EntityFabricationValidator` |
| `SELF_CONTRADICTION` | Response contradicts itself | `ConsistencyValidator` |
| `REPETITION` | Excessive phrase repetition | `RepetitionValidator` |
| `SEMANTIC_INCONSISTENCY` | Response unrelated to prompt | `SemanticValidator` |
| `TEMPORAL_INCONSISTENCY` | Temporal facts inconsistent | `TemporalConsistencyChecker` |
| `UNSUPPORTED_INFERENCE` | Unsupported logical inference | `AttributionChecker` |

## Usage

### Basic Usage

```cpp
#include "hallucination_harness.hpp"

using namespace qwen::hallucination;

// Configure harness
HarnessConfig config;
config.hallucination_threshold = 0.7f;
config.enable_repetition_check = true;
config.enable_numeric_check = true;

// Create harness and register validators
HallucinationHarness harness(config);
harness.register_validator(std::make_unique<FactualValidator>());
harness.register_validator(std::make_unique<NumericValidator>());
harness.register_validator(std::make_unique<ConsistencyValidator>());
harness.register_validator(std::make_unique<RepetitionValidator>());

// Validate a single prompt-response pair
auto report = harness.validate(
    "What is the capital of France?",
    "The capital of France is Paris."
);

// Check results
if (!report.passed) {
    std::cout << "Hallucination detected: " << report.hallucinations[0].description << "\n";
}
```

### Batch Validation

```cpp
std::vector<std::pair<std::string, std::string>> test_cases = {
    {"Prompt 1", "Response 1"},
    {"Prompt 2", "Response 2"},
    {"Prompt 3", "Response 3"}
};

auto reports = harness.batch_validate(test_cases);
```

### Custom Knowledge Base

```cpp
// Add verified facts programmatically
auto factual = std::make_unique<FactualValidator>();
factual->add_verified_fact("water boils at 100 celsius");
factual->add_verified_fact("earth is round");

// Or load from file
factual->load_facts_from_file("data/knowledge_base.txt");

harness.register_validator(std::move(factual));
```

## Running Tests

### Build

```bash
cd build
cmake .. -DENABLE_CUDA=OFF
cmake --build . --target harness_runner
```

### Run

```bash
./tests/hallucination_harness/harness_runner
```

## Output Format

### Console Output

```
========================================
HALLUCINATION VALIDATION REPORT
========================================

Prompt: What is 2 + 2?
Response: 2 + 2 = 5

Passed: false
Total Hallucinations: 1
Hallucination Score: 0.950
Validation Time: 2 ms

Detected Issues:
----------------------------------------
  [1] Numeric Inconsistency
      Confidence: 95.00%
      Description: Mathematical error detected
      Evidence: Expression "2+2=5" claims result is 5
```

### JSON Export

```json
{
  "total_tests": 7,
  "passed": 3,
  "average_score": 0.342,
  "results": [
    {
      "prompt": "...",
      "response": "...",
      "passed": false,
      "score": 0.95,
      "hallucination_count": 1,
      "validation_time_ms": 2
    }
  ]
}
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `hallucination_threshold` | 0.7 | Score above this = failed |
| `enable_factual_check` | true | Enable factual contradiction |
| `enable_numeric_check` | true | Enable numeric validation |
| `enable_consistency_check` | true | Enable self-consistency |
| `enable_repetition_check` | true | Enable repetition detection |
| `max_response_length` | 4096 | Max response length |
| `min_response_length` | 4 | Min response length |
| `repetition_threshold` | 0.3 | Repetition ratio threshold |
| `numeric_tolerance` | 1e-6 | Numeric comparison tolerance |

## Integration with Inference Engine

To integrate with your CUDA engine:

```cpp
// After inference
auto response = engine.generate(prompt);

// Validate before returning
auto report = harness.validate(prompt, response);

if (report.passed) {
    return response;
} else {
    // Log warning or regenerate
    log_warning("Hallucination detected: {}", report.hallucinations[0].description);
    return regenerate(prompt);
}
```