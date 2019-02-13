# -----------------------------------------------------------------------------------------------------
# Copyright (c) 2006-2019, Knut Reinert & Freie Universität Berlin
# Copyright (c) 2016-2019, Knut Reinert & MPI für molekulare Genetik
# This file may be used, modified and/or redistributed under the terms of the 3-clause BSD-License
# shipped with this file and also available at: https://github.com/seqan/seqan3/blob/master/LICENSE
# -----------------------------------------------------------------------------------------------------

cmake_minimum_required (VERSION 3.2...3.12)
project (example CXX)
message (STATUS "Configuring example for PIBF")

# ----------------------------------------------------------------------------
# Includes
# ----------------------------------------------------------------------------

include (CheckIncludeFileCXX)
include (CheckCXXSourceCompiles)

# ----------------------------------------------------------------------------
# Define paths
# ----------------------------------------------------------------------------

set(SEQAN_ROOT "${CMAKE_SOURCE_DIR}/submodules/seqan/")
set(SDSL_ROOT "${CMAKE_SOURCE_DIR}/submodules/sdsl-lite/")

# SeqAn include
set (CMAKE_INCLUDE_PATH ${SEQAN_ROOT}/include ${CMAKE_INCLUDE_PATH})

# SeqAn cmake module
set (CMAKE_PREFIX_PATH ${SEQAN_ROOT}/util/cmake ${CMAKE_PREFIX_PATH})
set (CMAKE_MODULE_PATH ${SEQAN_ROOT}/util/cmake ${CMAKE_MODULE_PATH})

include_directories (${CMAKE_INCLUDE_PATH})

# ----------------------------------------------------------------------------
# Require C++ Filesystem
# ----------------------------------------------------------------------------

# find the correct header
check_include_file_cxx (filesystem __HAVE_FILESYSTEM)
check_include_file_cxx (experimental/filesystem __HAVE_EXP_FILESYSTEM)

if (__HAVE_FILESYSTEM)
    message (STATUS "C++ Filesystem header:      <filesystem>")

    set (CXXSTD_TEST_SOURCE
        "#include <filesystem>
        int main()
        {
            std::filesystem::path p{\"\tmp/\"};
        }")
elseif (__HAVE_EXP_FILESYSTEM)
    message (STATUS "C++ Filesystem header:      <experimental/filesystem>")

    set (CXXSTD_TEST_SOURCE
        "#include <experimental/filesystem>
        int main()
        {
            std::experimental::filesystem::path p{\"/tmp/\"};
        }")
else ()
    message (FATAL_ERROR  "Requires C++17 filesystem support, but the filesystem header was not found.")
endif ()

# check if library is required
set (CMAKE_REQUIRED_LIBRARIES_ORIGINAL ${CMAKE_REQUIRED_LIBRARIES})

check_cxx_source_compiles ("${CXXSTD_TEST_SOURCE}" C++17FS_BUILTIN)

if (C++17FS_BUILTIN)
    message (STATUS "C++ Filesystem library:     builtin")
else ()
    set (C++17FS_LIB "")

    foreach (_LIB stdc++fs)
        set (CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES_ORIGINAL} ${_LIB})

        check_cxx_source_compiles ("${CXXSTD_TEST_SOURCE}" C++17FS_LIB-l${_LIB})

        if (C++17FS_LIB-l${_LIB})
            set (C++17FS_LIB ${_LIB})
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -l${_LIB}")
            break ()
        endif ()
        set (CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES_ORIGINAL})
    endforeach ()

    if (C++17FS_LIB)
        message (STATUS "C++ Filesystem library:     via -l${C++17FS_LIB}")
    else ()
        message (FATAL_ERROR  "Requires C++17 filesystem support, but your compiler does not offer it.")
    endif ()
endif ()

# ----------------------------------------------------------------------------
# Load SeqAn module
# ----------------------------------------------------------------------------

find_package (SeqAn REQUIRED)

# add include dir of test directory for some global test utility functions.
set (SEQAN_INCLUDE_DIRS "${SEQAN_INCLUDE_DIRS};${CMAKE_SOURCE_DIR}/include")

# import package's information (libraries are added with targets below)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SEQAN_CXX_FLAGS}")
include_directories (${SEQAN_INCLUDE_DIRS})
add_definitions (${SEQAN_DEFINITIONS})

# add warning levels
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Werror -Wall -Wextra -std=c++17")

# ----------------------------------------------------------------------------
# Use CCache if possible
# ----------------------------------------------------------------------------

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM}")
endif(CCACHE_PROGRAM)

# ----------------------------------------------------------------------------
# Add google/benchmark
# ----------------------------------------------------------------------------

set(BENCHMARK_ROOT "${PROJECT_BINARY_DIR}/vendor/benchmark")

include(ExternalProject)
set(
    google_benchmark_args
    "-DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    "-DBENCHMARK_ENABLE_TESTING=false"
)
ExternalProject_Add(
    google_benchmark
    PREFIX google_benchmark
    GIT_REPOSITORY "https://github.com/google/benchmark.git"
    SOURCE_DIR "${BENCHMARK_ROOT}"
    CMAKE_ARGS "${google_benchmark_args}"
    UPDATE_DISCONNECTED yes
)

add_library (benchmark STATIC IMPORTED)
add_dependencies(benchmark google_benchmark)
set_target_properties(benchmark PROPERTIES IMPORTED_LOCATION "${PROJECT_BINARY_DIR}/lib/libbenchmark.a")

# ----------------------------------------------------------------------------
# define all include folders
# ----------------------------------------------------------------------------

include_directories("${BENCHMARK_ROOT}/include/")
include_directories("${SEQAN_ROOT}/include/")
include_directories("${SDSL_ROOT}/include/")

# ----------------------------------------------------------------------------
# Add the benchmarks
# ----------------------------------------------------------------------------

add_definitions(-DBASE_DIR="${CMAKE_SOURCE_DIR}")
add_executable(example example.cpp)
target_link_libraries(example "pthread" "benchmark")
add_dependencies(example benchmark)
