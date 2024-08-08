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

enable_language(CUDA)
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

file(GLOB CUPTI_SOURCES *.cu)

foreach (INPUT_FILE ${CUPTI_SOURCES})
    get_filename_component(OUTPUT_FILE ${INPUT_FILE} NAME_WE)
    set(OUTPUT_FILE "cupti_${OUTPUT_FILE}")

    add_executable(${OUTPUT_FILE} ${INPUT_FILE})
    # no pass arquitecture flags to compiler
    set_property(TARGET ${OUTPUT_FILE} PROPERTY CUDA_ARCHITECTURES 89)

    target_link_libraries (${OUTPUT_FILE} argparser CUDA::cupti)
endforeach()
