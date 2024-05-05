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

file(GLOB CUDA_SOURCES *.cu)

foreach (INPUT_FILE ${CUDA_SOURCES})
    get_filename_component(OUTPUT_FILE ${INPUT_FILE} NAME_WE)
    add_executable(${OUTPUT_FILE} ${INPUT_FILE})
    # no pass arquitecture flags to compiler
    set_property(TARGET ${OUTPUT_FILE} PROPERTY CUDA_ARCHITECTURES OFF)

    target_link_libraries (${OUTPUT_FILE} argparser)
endforeach()

# Cuda tests ===============
foreach(X 32 64 512 2048 131072 4194304 67108864)
    # These need to be a power of two
    add_test(NAME "cuda_bitonicsort_${X}" COMMAND $<TARGET_FILE:bitonicsort> ${X})
endforeach()

foreach(X 32 64 69 512 600 2048 131072 4194304 67108864)
    add_test(NAME "cuda_count_${X}" COMMAND $<TARGET_FILE:count> ${X})
endforeach()

foreach(X 32 64 69 512 600 2048 131072 4194304 67108864)
    add_test(NAME "cuda_reduction_${X}" COMMAND $<TARGET_FILE:reduction> ${X})
endforeach()

foreach(X 32 64 69 512 600 2048 131072 4194304 67108864)
    add_test(NAME "cuda_scan_${X}" COMMAND $<TARGET_FILE:scan> ${X})
endforeach()

