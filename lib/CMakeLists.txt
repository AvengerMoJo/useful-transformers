find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

set(SOURCE_FILES
  attention.h
  bias.h
  conv.h
  fp16.h
  layernorm.h
  matmul.h
  matmul_backend.h
  misc.h
  residual_attention_block.h
  softmax.h
  profiler.h
  attention.cc
  bias.cc
  matmul.cc
  matmul_cpu.cc
  residual_attention_block.cc
  softmax.cc
  profiler.cc
  conv.cc
  layernorm.cc
)

if (RKNN_BACKEND)
  list(APPEND SOURCE_FILES rknn_api.h rknn_matmul_api.h matmul_rknn.cc)
endif()

add_library(ctransformer SHARED STATIC ${SOURCE_FILES})
if (RKNN_BACKEND)
  target_link_libraries(ctransformer
    PRIVATE
    "${PROJECT_SOURCE_DIR}/lib/librknnrt.so"
  )
endif()

pybind11_add_module(pybind_residual_attention_block
  MODULE pybind_residual_attention_block.cc
)
pybind11_extension(pybind_residual_attention_block)
target_link_libraries(pybind_residual_attention_block
  PRIVATE ctransformer
)

if (RKNN_BACKEND)
  install(FILES "${PROJECT_SOURCE_DIR}/lib/librknnrt.so" DESTINATION .)
endif()
