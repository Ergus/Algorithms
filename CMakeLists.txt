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

project(ArgParserC LANGUAGES CXX CUDA)

set(CMAKE_VERBOSE_MAKEFILE on)

include_directories(h hpp)

add_subdirectory("submodules/ArgParserC" EXCLUDE_FROM_ALL)
include_directories("submodules/ArgParserC")

file(GLOB CPP_SOURCES cpp/*.cpp)
file(GLOB CPP_SOURCES cuda/*.cu)

enable_testing()

foreach (INPUT_FILE ${CPP_SOURCES})
    get_filename_component(OUTPUT_FILE ${INPUT_FILE} NAME_WE)
    add_executable(${OUTPUT_FILE} ${INPUT_FILE})
    target_link_libraries (${OUTPUT_FILE} argparser)

    add_test(NAME ${OUTPUT_FILE} COMMAND ${OUTPUT_FILE} 40)
endforeach()
