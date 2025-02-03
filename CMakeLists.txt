# Copyright (C) 2024  Jimmy Aguilar Mena

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


cmake_minimum_required(VERSION 3.10)

project(Algorithms LANGUAGES CXX)

include(CheckLanguage)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_STANDARD 20)

include_directories(h hpp)

add_subdirectory("submodules/ArgParserC" EXCLUDE_FROM_ALL)
include_directories("submodules/ArgParserC")

file(GLOB CPP_SOURCES cpp/*.cpp)

foreach (INPUT_FILE ${CPP_SOURCES})
    get_filename_component(OUTPUT_FILE ${INPUT_FILE} NAME_WE)
    add_executable(${OUTPUT_FILE} ${INPUT_FILE})
    target_link_libraries (${OUTPUT_FILE} argparser)

    target_compile_options(${OUTPUT_FILE} PRIVATE
	$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wall -Wextra -pedantic -Werror -Wno-ignored-qualifiers>
	$<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>)

endforeach()

enable_testing()

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    add_subdirectory(cuda)

    # Check if cupti is available.
    find_package(CUDAToolkit COMPONENTS cupti)
    if (TARGET CUDA::cupti)
	add_subdirectory(cupti)
    endif()

else()
    message(STATUS "No Cuda support")
endif()
# --------------------------------------------------------------------
# C++ test
foreach(X 32 64 65 512 513 2048 131072 4194304)
    # These need to be a power of two
    add_test(NAME "cpp_mergesort_${X}" COMMAND $<TARGET_FILE:mergesort> ${X})
    add_test(NAME "cpp_heapsort_${X}" COMMAND $<TARGET_FILE:heapsort> ${X})
    add_test(NAME "cpp_quicksort_${X}" COMMAND $<TARGET_FILE:quicksort> ${X})
endforeach()

foreach(X 512 2048 2050 2090 2091 131072 4194304)
    # These need to be a power of two
    add_test(NAME "cpp_parallelscan_${X}" COMMAND $<TARGET_FILE:parallelscan> ${X})
endforeach()

add_test(NAME "cpp_rwlock_1" COMMAND $<TARGET_FILE:rwlock> 10 100 10 100)
add_test(NAME "cpp_rwlock_2" COMMAND $<TARGET_FILE:rwlock> 10 1000 10 100)
add_test(NAME "cpp_rwlock_3" COMMAND $<TARGET_FILE:rwlock> 10 100 10 1000)
add_test(NAME "cpp_rwlock_4" COMMAND $<TARGET_FILE:rwlock> 10 1000 10 1000)

add_test(NAME "cpp_rwlock2_1" COMMAND $<TARGET_FILE:rwlock2> 10 100 10 100)
add_test(NAME "cpp_rwlock2_2" COMMAND $<TARGET_FILE:rwlock2> 10 1000 10 100)
add_test(NAME "cpp_rwlock2_3" COMMAND $<TARGET_FILE:rwlock2> 10 100 10 1000)
add_test(NAME "cpp_rwlock2_4" COMMAND $<TARGET_FILE:rwlock2> 10 1000 10 1000)

add_test(NAME "cpp_lrucache_10" COMMAND $<TARGET_FILE:lrucache> 10)
add_test(NAME "cpp_lrucache_20" COMMAND $<TARGET_FILE:lrucache> 20)
add_test(NAME "cpp_lrucache_40" COMMAND $<TARGET_FILE:lrucache> 40)

add_test(NAME "cpp_queuelock_10" COMMAND $<TARGET_FILE:queuelock> 10)
add_test(NAME "cpp_queuelock_20" COMMAND $<TARGET_FILE:queuelock> 20)
add_test(NAME "cpp_queuelock_40" COMMAND $<TARGET_FILE:queuelock> 40)

add_test(NAME "cpp_fair_lock_10" COMMAND $<TARGET_FILE:fairlock> 10)
add_test(NAME "cpp_fair_lock_20" COMMAND $<TARGET_FILE:fairlock> 20)
add_test(NAME "cpp_fair_lock_40" COMMAND $<TARGET_FILE:fairlock> 40)

add_test(NAME "cpp_range_map" COMMAND $<TARGET_FILE:rangemap>)
add_test(NAME "cpp_contiguous_map" COMMAND $<TARGET_FILE:contiguousmap>)
add_test(NAME "cpp_matrix" COMMAND $<TARGET_FILE:matrix>)

add_test(NAME "cpp_vrange_2_8_2" COMMAND $<TARGET_FILE:vrange> 2 8 2)
add_test(NAME "cpp_vrange_8_2_-1" COMMAND $<TARGET_FILE:vrange> 8 2 -2)
add_test(NAME "cpp_vrange_0_10_5" COMMAND $<TARGET_FILE:vrange> 0 10 5)
add_test(NAME "cpp_vrange_0_10_1" COMMAND $<TARGET_FILE:vrange> 0 10 1)
