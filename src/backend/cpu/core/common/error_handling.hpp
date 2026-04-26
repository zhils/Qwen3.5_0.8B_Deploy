#pragma once

#include <stdexcept>
#include <string>
#include <sstream>
#include <cstring>

namespace qwen {

// ============================================================
// Custom Exception Types
// ============================================================

class InferenceError : public std::runtime_error {
  public:
    explicit InferenceError(const std::string& message)
        : std::runtime_error(message) {}

    InferenceError(const std::string& component, const std::string& message)
        : std::runtime_error("[" + component + "] " + message) {}
};

class WeightError : public InferenceError {
  public:
    explicit WeightError(const std::string& message)
        : InferenceError("WeightLoader", message) {}
};

class DimensionError : public InferenceError {
  public:
    explicit DimensionError(const std::string& message)
        : InferenceError("Dimension", message) {}

    DimensionError(const std::string& expected, const std::string& actual)
        : InferenceError("Dimension",
                         "Size mismatch: expected " + expected + ", got " + actual) {}
};

class StateError : public InferenceError {
  public:
    explicit StateError(const std::string& message)
        : InferenceError("State", message) {}
};

class FileError : public InferenceError {
  public:
    explicit FileError(const std::string& filename, const std::string& message)
        : InferenceError("File[" + filename + "]", message) {}
};

// ============================================================
// Validation Macros
// ============================================================

#define QWEN_CHECK(cond, msg)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            throw qwen::InferenceError("Check", msg);                                              \
        }                                                                                          \
    } while (0)

#define QWEN_CHECK_DIM(expected, actual)                                                           \
    do {                                                                                           \
        if ((expected) != (actual)) {                                                              \
            throw qwen::DimensionError(std::to_string(expected), std::to_string(actual));          \
        }                                                                                          \
    } while (0)

#define QWEN_CHECK_NOT_NULL(ptr, name)                                                             \
    do {                                                                                           \
        if ((ptr) == nullptr) {                                                                    \
            throw qwen::InferenceError("NullCheck", std::string(name) + " is null");               \
        }                                                                                          \
    } while (0)

#define QWEN_CHECK_POSITIVE(value, name)                                                           \
    do {                                                                                           \
        if ((value) <= 0) {                                                                        \
            throw qwen::DimensionError(std::string(name) + " must be positive, got " +             \
                                       std::to_string(value));                                     \
        }                                                                                          \
    } while (0)

#define QWEN_CHECK_NOT_EMPTY(container, name)                                                      \
    do {                                                                                           \
        if ((container).empty()) {                                                                 \
            throw qwen::StateError(std::string(name) + " is not initialized");                     \
        }                                                                                          \
    } while (0)

#define QWEN_CHECK_WEIGHT_SIZE(expected, actual, name)                                             \
    do {                                                                                           \
        if ((expected) != (actual)) {                                                              \
            std::ostringstream oss;                                                                \
            oss << name << " weight size mismatch: expected " << (expected) << ", got "            \
                << (actual);                                                                       \
            throw qwen::WeightError(oss.str());                                                    \
        }                                                                                          \
    } while (0)

// ============================================================
// Utility Functions
// ============================================================

inline std::string format_dims(const char* name, int expected, int actual) {
    std::ostringstream oss;
    oss << name << " dimension mismatch: expected " << expected << ", got " << actual;
    return oss.str();
}

inline std::string format_weight_info(const char* name, size_t expected, size_t actual) {
    std::ostringstream oss;
    oss << name << " weight size mismatch: expected " << expected << ", got " << actual;
    return oss.str();
}

} // namespace qwen
