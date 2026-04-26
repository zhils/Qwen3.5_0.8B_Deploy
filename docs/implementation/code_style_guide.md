# Qwen3.5-0.8B C++/CUDA 代码规范

本文档定义项目的代码规范和最佳实践，所有贡献者应遵循这些规范。

## 1. 命名规范

### 1.1 文件命名
- **头文件**: `.hpp` 扩展名
- **源文件**: `.cpp` 扩展名
- **CUDA 头文件**: `.cuh` 扩展名
- **CUDA 源文件**: `.cu` 扩展名
- **文件名**: 小写字母 + 下划线分隔 (snake_case)

```
✅ language_common.hpp
✅ token_embedding.cpp
✅ cuda_engine.cuh
✅ rmsnorm_cuda.cu

❌ LanguageCommon.h
❌ TokenEmbedding.cc
```

### 1.2 类命名
- 使用 PascalCase (大驼峰)
- 名词或名词短语

```cpp
✅ class TokenEmbedding
✅ class FullAttention
✅ class LinearAttention
✅ struct KVCache

❌ class token_embedding
❌ class full_attention
```

### 1.3 函数命名
- 使用 snake_case (小写 + 下划线)
- 动词或动词短语

```cpp
✅ void set_weights(...)
✅ std::vector<float> forward(...)
✅ void check_ready() const
✅ float compute_attention(...)

❌ void SetWeights(...)
❌ std::vector<float> Forward(...)
```

### 1.4 变量命名
- **成员变量**: snake_case + 后缀 `_`
- **局部变量**: snake_case
- **常量**: UPPER_SNAKE_CASE
- **模板参数**: PascalCase

```cpp
class Example {
  private:
    int hidden_size_;        // 成员变量
    float eps_;              // 成员变量
    std::vector<float> weight_; // 成员变量

    static constexpr int MAX_SEQ_LEN = 262144;  // 常量
};

void function() {
    int batch_size = 32;     // 局部变量
    float learning_rate = 0.001f; // 局部变量
}
```

### 1.5 命名空间
- 小写字母
- 项目使用 `qwen` 作为主命名空间
- CUDA 相关使用 `qwen::cuda` 子命名空间

```cpp
namespace qwen {
    class TokenEmbedding { ... };
}

namespace qwen {
namespace cuda {
    void cuda_check(...) { ... };
}
}
```

## 2. 代码格式

### 2.1 缩进
- 使用 4 个空格缩进
- 不使用 Tab 字符
- 由 `.clang-format` 自动格式化

### 2.2 行长度
- 最大行长度: 120 字符
- 长表达式应适当换行

### 2.3 大括号
- 使用 Attach 风格 (与函数声明同行)
- 控制语句大括号始终存在

```cpp
// ✅ 正确
void function() {
    if (condition) {
        do_something();
    } else {
        do_other();
    }
}

// ❌ 错误
void function()
{
    if (condition)
        do_something();
}
```

### 2.4 指针和引用
- 指针和引用符号靠近类型

```cpp
✅ int* ptr;
✅ const float& ref;
✅ void function(const std::vector<float>& input);

❌ int *ptr;
❌ const float &ref;
```

### 2.5 空行
- 函数之间保留一个空行
- 逻辑块之间保留一个空行
- 最多保留一个连续空行

## 3. 错误处理

### 3.1 使用统一的错误处理宏

项目提供了一组错误处理宏，位于 `error_handling.hpp`:

```cpp
#include "error_handling.hpp"

// 检查条件
QWEN_CHECK(condition, "error message");

// 检查维度匹配
QWEN_CHECK_DIM(expected, actual);

// 检查指针非空
QWEN_CHECK_NOT_NULL(ptr, "name");

// 检查正数
QWEN_CHECK_POSITIVE(value, "name");

// 检查容器非空
QWEN_CHECK_NOT_EMPTY(container, "name");

// 检查权重尺寸
QWEN_CHECK_WEIGHT_SIZE(expected, actual, "name");
```

### 3.2 自定义异常类型

```cpp
// 通用推理错误
throw InferenceError("component", "message");

// 权重相关错误
throw WeightError("message");

// 维度不匹配错误
throw DimensionError("expected", "actual");

// 状态错误
throw StateError("message");

// 文件错误
throw FileError("filename", "message");
```

### 3.3 错误处理示例

```cpp
void RMSNorm::set_weight(std::vector<float> weight) {
    QWEN_CHECK_WEIGHT_SIZE(hidden_size_, weight.size(), "RMSNorm");
    weight_ = std::move(weight);
}

std::vector<float> RMSNorm::forward(const std::vector<float>& input) const {
    check_ready();
    QWEN_CHECK_DIM(hidden_size_, input.size());
    
    // ... 计算逻辑
}

void RMSNorm::forward_batch(const float* input, float* output, int batch_size) const {
    check_ready();
    QWEN_CHECK_NOT_NULL(input, "input");
    QWEN_CHECK_NOT_NULL(output, "output");
    QWEN_CHECK_POSITIVE(batch_size, "batch_size");
    
    // ... 计算逻辑
}
```

## 4. 头文件规范

### 4.1 Include Guards
- 使用 `#pragma once`

### 4.2 Include 顺序
1. 对应的头文件 (如果是 .cpp)
2. 项目头文件
3. 标准库头文件
4. 第三方库头文件

每组之间用空行分隔，组内按字母顺序排列。

```cpp
// token_embedding.cpp
#include "token_embedding.hpp"
#include "error_handling.hpp"

#include <algorithm>
#include <vector>
```

### 4.3 头文件最小依赖
- 只在头文件中 include 必需的头文件
- 使用前向声明减少依赖

```cpp
// ✅ 使用前向声明
class TokenEmbedding;

// ❌ 不必要的 include
#include "token_embedding.hpp"
```

## 5. 类设计

### 5.1 访问修饰符顺序
```cpp
class Example {
  public:
    // 公共接口
    
  protected:
    // 受保护成员
    
  private:
    // 私有实现
};
```

### 5.2 构造函数
- 使用初始化列表
- 在构造函数中验证参数

```cpp
TokenEmbedding::TokenEmbedding(int vocab_size, int hidden_size)
    : vocab_size_(vocab_size), hidden_size_(hidden_size) {
    QWEN_CHECK_POSITIVE(vocab_size, "vocab_size");
    QWEN_CHECK_POSITIVE(hidden_size, "hidden_size");
}
```

### 5.3 const 正确性
- 不修改成员状态的函数标记为 `const`
- 使用 `const&` 传递大对象

```cpp
class Example {
  public:
    int size() const { return size_; }
    void process(const std::vector<float>& input);
    
  private:
    void check_ready() const;
};
```

## 6. CUDA 代码规范

### 6.1 CUDA 错误检查
- 始终使用 `CUDA_CHECK` 宏检查 CUDA API 调用
- 使用 `CUDA_CHECK_LAST_KERNEL()` 检查 kernel 执行

```cpp
#include "cuda_error_handling.cuh"

void cuda_function() {
    float* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, size * sizeof(float)));
    
    kernel<<<blocks, threads>>>(d_ptr, size);
    CUDA_CHECK_LAST_KERNEL();
}
```

### 6.2 内存管理
- 使用 RAII 管理 GPU 内存
- 及时释放不再使用的 GPU 内存

### 6.3 Kernel 设计
- 使用有意义的 kernel 名称
- 添加适当的边界检查

## 7. 测试规范

### 7.1 测试文件命名
- 单元测试: `test_<module>.cpp`
- 集成测试: 描述性名称

### 7.2 测试结构
```cpp
// 1. 设置测试环境
// 2. 执行测试
// 3. 验证结果
// 4. 清理资源
```

## 8. 注释规范

### 8.1 文件头注释
```cpp
/**
 * @file token_embedding.hpp
 * @brief Token embedding layer implementation
 * 
 * Converts token IDs to dense vector representations.
 */
```

### 8.2 函数注释
```cpp
/**
 * @brief Compute forward pass
 * @param input Input tensor
 * @return Output tensor
 * @throws DimensionError if input size mismatch
 */
std::vector<float> forward(const std::vector<float>& input) const;
```

### 8.3 内联注释
- 解释"为什么"而不是"做什么"
- 保持简洁明了

## 9. Git 提交规范

### 9.1 提交消息格式
```
<type>: <description>

[optional body]
```

### 9.2 Type 类型
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式
- `refactor`: 重构
- `test`: 测试相关
- `perf`: 性能优化

### 9.3 示例
```
feat: add INT8 KV cache benchmark
fix: resolve dimension mismatch in linear attention
docs: update API documentation
refactor: unify error handling across modules
```

## 10. 构建和测试

### 10.1 本地构建
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### 10.2 运行测试
```bash
cd build
# Windows (多配置生成器)
ctest -C Release --output-on-failure

# Linux/macOS
ctest --output-on-failure
```

### 10.3 代码格式化
```bash
clang-format -i src/**/*.cpp src/**/*.hpp
```

## 11. 性能优化建议

### 11.1 内存优化
- 使用 `reserve()` 预分配 vector 容量
- 使用 `std::move()` 避免不必要的拷贝
- 优先使用引用传递

### 11.2 计算优化
- 避免在循环中重复计算
- 使用缓存友好的数据访问模式
- 考虑使用 SIMD 指令

### 11.3 CUDA 优化
- 合理使用共享内存
- 避免 warp divergence
- 优化内存访问模式

## 12. 代码审查清单

提交 PR 前检查:
- [ ] 代码通过 clang-format 格式化
- [ ] 所有测试通过
- [ ] 添加了适当的错误处理
- [ ] 更新了相关文档
- [ ] 提交消息符合规范
- [ ] 没有留下调试代码
- [ ] 没有硬编码路径

---

最后更新: 2026-04-20
