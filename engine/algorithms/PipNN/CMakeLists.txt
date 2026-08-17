# RSQ kernel library (8-bit rotated scalar quantization + derived 1-bit panel).
# NOTE: These .cc files use HWY's foreach_target model which requires them
# to be compiled as a separate translation unit. CMake properly tracks
# header dependencies, so changes to .h files will trigger recompilation.
add_library(rsq_kernels
  quantization/rsq_encoder.cc
  quantization/pipnn_rsq.cc
  quantization/assignleaders_rsq.cc
  quantization/processleaf_rsq.cc
)
# IMPORTANT: the `quantization` entry must stay FIRST. Highway's foreach_target
# re-inclusion uses bare filenames (HWY_TARGET_INCLUDE "processleaf_rsq.cc"),
# which only resolve through this include directory.
# IMPORTANT: Include ../bench so the library uses the SAME local parlay headers
# as the main binary (bench/parlay/* -> ../../parlaylib/include/parlay). Do NOT
# link against the `parlay` CMake target here — that would add a second parlay
# include path and risk two different parlay implementations in one binary (ODR).
target_include_directories(rsq_kernels PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/quantization
  ${CMAKE_CURRENT_SOURCE_DIR}/../bench
)
target_link_libraries(rsq_kernels PUBLIC
  Eigen3::Eigen
  absl::status
  absl::statusor
  absl::span
  absl::function_ref
  absl::log
  absl::check
  hwy
  hwy_contrib
)
target_compile_features(rsq_kernels PRIVATE cxx_std_20)

# --- The submission binary ---------------------------------------------------
# The engine is built in exactly one variant: float input x MIPS distance.
set(TARGET_NAME "neighbors-pipnn_FLOAT_T_MIPS")

add_executable(${TARGET_NAME} ../bench/neighborsTime.C)

target_link_libraries(${TARGET_NAME} PRIVATE parlay ParlayANN
  Eigen3::Eigen absl::synchronization absl::flat_hash_set
  absl::span hwy hwy_contrib rsq_kernels mimalloc-static)
target_include_directories(${TARGET_NAME} PRIVATE ../bench)
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
# LOAD-BEARING: ../bench/neighborsTime.C does NOT #include neighbors.h. It is
# force-included through this precompiled header, which is the only way ANN<>()
# reaches main(). Removing this line breaks the build with no obvious cause;
# "fixing" it with an explicit #include changes template instantiation
# behaviour and is not what the submitted binary was built from.
target_precompile_headers(${TARGET_NAME} PRIVATE neighbors.h)

message(STATUS "Configured Target: ${TARGET_NAME}")
