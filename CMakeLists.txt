cmake_minimum_required(VERSION 3.18)
project(qwen_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)
option(ENABLE_WEIGHT_DEPENDENT_TESTS "Enable tests that require model weights" OFF)
include(CTest)
enable_testing()

set(CORE_SOURCES
    src/backend/cpu/core/common/language_common.cpp
    src/backend/cpu/core/common/language_backbone.cpp
    src/backend/cpu/core/attention/language_linear_attn.cpp
    src/backend/cpu/core/attention/language_full_attn.cpp
    src/backend/cpu/core/mlp/language_mlp.cpp
    src/backend/cpu/core/embedding/token_embedding.cpp
    src/backend/cpu/core/embedding/multimodal_embedding.cpp
    src/backend/cpu/core/heads/lm_head.cpp
    src/backend/cpu/core/heads/sampler.cpp
    src/backend/cpu/core/heads/mtp_head.cpp
)

set(VISION_SOURCES
    src/backend/cpu/vision/vision_patch_embedding.cpp
    src/backend/cpu/vision/vision_transformer.cpp
    src/backend/cpu/vision/qwen35_image_preprocess.cpp
)

add_library(qwen_core STATIC ${CORE_SOURCES} ${VISION_SOURCES})
target_include_directories(qwen_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src/backend/cpu/core
    ${CMAKE_SOURCE_DIR}/src/backend/cpu/core/common
    ${CMAKE_SOURCE_DIR}/src/backend/cpu/core/attention
    ${CMAKE_SOURCE_DIR}/src/backend/cpu/core/embedding
    ${CMAKE_SOURCE_DIR}/src/backend/cpu/core/heads
    ${CMAKE_SOURCE_DIR}/src/backend/cpu/core/mlp
    ${CMAKE_SOURCE_DIR}/src/backend/cpu/vision
    ${CMAKE_SOURCE_DIR}/third_party
)

if(MSVC)
    target_compile_options(qwen_core PRIVATE /W4 /O2)
    target_link_options(qwen_core PRIVATE /STACK:268435456)
else()
    target_compile_options(qwen_core PRIVATE -Wall -Wextra -O2)
endif()

add_executable(e2e_inference tests/integration/e2e_inference_test.cpp)
target_link_libraries(e2e_inference PRIVATE qwen_core)
if(MSVC)
    target_link_options(e2e_inference PRIVATE /STACK:268435456)
endif()

add_executable(test_rmsnorm tests/unit/test_rmsnorm.cpp)
target_link_libraries(test_rmsnorm PRIVATE qwen_core)

add_executable(test_backbone tests/unit/test_backbone.cpp)
target_link_libraries(test_backbone PRIVATE qwen_core)

add_executable(test_full_attn tests/unit/test_full_attn.cpp)
target_link_libraries(test_full_attn PRIVATE qwen_core)

add_executable(test_linear_attn tests/unit/test_linear_attn.cpp)
target_link_libraries(test_linear_attn PRIVATE qwen_core)

add_executable(test_kvcache tests/unit/test_kvcache.cpp)
target_link_libraries(test_kvcache PRIVATE qwen_core)

if(BUILD_TESTING)
    add_test(NAME unit_kvcache COMMAND test_kvcache)
    if(ENABLE_WEIGHT_DEPENDENT_TESTS)
        add_test(NAME unit_rmsnorm COMMAND test_rmsnorm)
        add_test(NAME unit_backbone COMMAND test_backbone)
        add_test(NAME unit_full_attn COMMAND test_full_attn)
        add_test(NAME unit_linear_attn COMMAND test_linear_attn)
    endif()
endif()

add_executable(cpu_benchmark tests/integration/cpu_benchmark.cpp)
target_link_libraries(cpu_benchmark PRIVATE qwen_core)

add_executable(stage2_cpu_benchmark tests/integration/stage2_benchmark.cpp)
target_link_libraries(stage2_cpu_benchmark PRIVATE qwen_core)
if(WIN32)
    target_link_libraries(stage2_cpu_benchmark PRIVATE psapi)
endif()
if(MSVC)
    target_link_options(stage2_cpu_benchmark PRIVATE /STACK:268435456)
endif()

if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_package(CUDAToolkit REQUIRED)

    set(CUDA_KERNEL_SOURCES
        src/backend/cuda/kernels/rmsnorm_cuda.cu
        src/backend/cuda/kernels/mlp_cuda.cu
        src/backend/cuda/kernels/full_attention_cuda.cu
        src/backend/cuda/kernels/linear_attention_cuda.cu
        src/backend/cuda/kernels/lm_head_cuda.cu
        src/backend/cuda/kernels/token_embedding_cuda.cu
        src/backend/cuda/kernels/gpu_sampler_argmax.cu
        src/backend/cuda/kernels/cuda_engine.cu
        src/backend/cuda/kernels/flash_attention.cu
        src/backend/cuda/kernels/vision_patch_embedding_cuda.cu
        src/backend/cuda/kernels/fa_wrapper.cpp
        src/backend/cuda/kernels/fa_set_weights.cu
        src/backend/cuda/kernels/fused_kernels.cu
    )

    add_library(qwen_cuda STATIC ${CUDA_KERNEL_SOURCES})
    target_include_directories(qwen_cuda PUBLIC
        ${CMAKE_SOURCE_DIR}/src/backend/cuda/include
        ${CMAKE_SOURCE_DIR}/src/backend/cpu/core
        ${CMAKE_SOURCE_DIR}/src/backend/cpu/core/common
        ${CMAKE_SOURCE_DIR}/src/backend/cpu/core/attention
        ${CMAKE_SOURCE_DIR}/src/backend/cpu/core/embedding
        ${CMAKE_SOURCE_DIR}/src/backend/cpu/core/heads
        ${CMAKE_SOURCE_DIR}/src/backend/cpu/core/mlp
        ${CMAKE_SOURCE_DIR}/src/backend/cpu/vision
    )
    target_link_libraries(qwen_cuda PUBLIC
        qwen_core
        CUDA::cudart
        CUDA::cublas
    )
    set_target_properties(qwen_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
    )

    add_executable(performance_test src/backend/cuda/performance_test.cu)
    target_link_libraries(performance_test PRIVATE qwen_cuda)
endif()
