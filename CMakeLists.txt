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


cmake_minimum_required(VERSION 3.5)

project(AlgorithmMisc LANGUAGES CXX CUDA)

set(CMAKE_VERBOSE_MAKEFILE on)

include_directories(h hpp)

add_subdirectory("submodules/ArgParserC" EXCLUDE_FROM_ALL)
include_directories("submodules/ArgParserC")

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory("submodules/googletest" EXCLUDE_FROM_ALL)
include_directories("submodules/googletest/googletest/include")

file(GLOB CPP_SOURCES cpp/*.cpp)
file(GLOB CUDA_SOURCES cuda/*.cu)

set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

enable_testing()

foreach (INPUT_FILE ${CPP_SOURCES} ${CUDA_SOURCES})
    get_filename_component(OUTPUT_FILE ${INPUT_FILE} NAME_WE)
    add_executable(${OUTPUT_FILE} ${INPUT_FILE})
    # no pass arquitecture flags to compiler
    set_property(TARGET ${OUTPUT_FILE} PROPERTY CUDA_ARCHITECTURES OFF)

    target_link_libraries (${OUTPUT_FILE} argparser)
endforeach()
