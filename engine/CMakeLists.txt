cmake_minimum_required(VERSION 3.15)
project(ParlayANN VERSION 1
        DESCRIPTION "PiPNN all-kNN graph engine — SISAP 2026 Task 1 submission (minimal tree)."
        LANGUAGES CXX)

include(CheckCXXCompilerFlag)
include (FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
# Pin znver2 (AVX2, no AVX-512) instead of -march=native so the emitted code
# does not depend on the build host. Highway does its own runtime ISA dispatch,
# so its kernels are unaffected; -march only governs non-Highway code.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=znver2 -mtune=znver2 -ffast-math")

# Do not silently re-fetch pinned dependencies on reconfigure.
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# --- ParlayLib ---------------------------------------------------------------
# ParlayLib is vendored under parlaylib/ (see parlaylib/PROVENANCE.md), reached
# through the algorithms/bench/parlay symlink; this interface target supplies
# its usage requirements. Do NOT add a second parlaylib copy (ODR hazard).
add_library(parlay INTERFACE)
target_include_directories(parlay INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/parlaylib/include)
target_compile_features(parlay INTERFACE cxx_std_17)
find_package(Threads REQUIRED)
target_link_libraries(parlay INTERFACE Threads::Threads)
target_compile_definitions(parlay INTERFACE PARLAY_ELASTIC_PARALLELISM=true)

# --- Pinned third-party dependencies -----------------------------------------
# Every GIT_TAG below is the exact commit that produced the submitted binary;
# pinning them is required for a reproducible build.
FetchContent_Declare(Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3147391d946bb4b6c68edd901f2add6ac1f31f8c  # 3.4.0
)
FetchContent_MakeAvailable(Eigen)

FetchContent_Declare(abseil-cpp
  GIT_REPOSITORY  https://github.com/abseil/abseil-cpp.git
  GIT_TAG         bc257a88f7c1939f24e0379f14a3589e926c950c  # 20250512.0
)
FetchContent_MakeAvailable(abseil-cpp)

FetchContent_Declare(highway
  GIT_REPOSITORY  https://github.com/google/highway/
  GIT_TAG         eb3479ebfceac4e2986dce3f902baa086904c4f9  # was: master
)
FetchContent_MakeAvailable(highway)

FetchContent_Declare(mimalloc
  GIT_REPOSITORY  https://github.com/microsoft/mimalloc.git
  GIT_TAG         f81bf1b31af819a31195e08f9546dc80f8931587  # v2.2.2
)
set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(MI_BUILD_OBJECT OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(mimalloc)

# Set module path
list(APPEND CMAKE_MODULE_PATH "${ParlayANN_SOURCE_DIR}/cmake")

# HDF5 — needed for direct HDF5 reads in PointRange (SISAP 2026).
find_package(HDF5 REQUIRED COMPONENTS C)

add_library(ParlayANN INTERFACE)
set(ParlayANN_INCLUDE_DIR "${ParlayANN_SOURCE_DIR}/algorithms")
target_include_directories(ParlayANN INTERFACE
  $<BUILD_INTERFACE:${ParlayANN_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${HDF5_INCLUDE_DIRS})
target_link_libraries(ParlayANN INTERFACE ${HDF5_C_LIBRARIES})

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(algorithms)
