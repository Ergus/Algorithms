// Copyright (C) 2025  Jimmy Aguilar Mena

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

float myGetElapsed(const cudaEvent_t &eStart, const cudaEvent_t &eStop)
{
	cudaEventSynchronize(eStop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, eStart, eStop);
	return milliseconds;
}

class cudaErrorCheck {
public:
	cudaErrorCheck &operator=(cudaError_t err)
	{
		if (err != cudaSuccess) {
			cudaDeviceSynchronize();
			printf("CUDA Error: %s\n", cudaGetErrorString(err));
			abort();
		}
		return *this;
	}
};


template <typename T>
__device__ T __unit(const T& a)
{
	return a;
}

template <typename T>
__device__ T __sum(const T& a,const T& b)
{
	return a + b;
}

/**
	This was expected to be available, but it is not
 **/
template <typename T>
__device__ void swap(T &v1, T &v2)
{
	T tmp = v1;
	v1 = v2;
	v2 = tmp;
}

template <typename TFunc>
std::pair<dim3, dim3> get_grid_block_dims(
	size_t array_size,
	TFunc func,
	int dynamic_shared_mem,
	int  block_size_limit = 0
) {
	cudaErrorCheck err;

	// Compute max concurrent occupancy combination
	// Compute blockSize; The grid size from here may not be accurate enough because
	// we need to use the sharedSize to get the accurate value for numBlocksPerSm
	int minGridSize = 0, blockSize = 0;
	err = cudaOccupancyMaxPotentialBlockSize(
		&minGridSize,
		&blockSize,
		func,
		dynamic_shared_mem,
		block_size_limit
	);
	assert(blockSize > 0);
	assert(minGridSize > 0);

	// Now get the SM x Blocks_per_sm  ==================

	// Compute how many blocks fit in an SM
	int numBlocksPerSm = 0;
	err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&numBlocksPerSm,
		func,
		blockSize,
		0
	);

	// Now count how many SMs I have
	int numSMs;
    err = cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

	if (minGridSize != numSMs * numBlocksPerSm) {
		fprintf(stderr, "minGridSize(%d) != numSMs(%d) * numBlocksPerSm(%d)\n",
			minGridSize, numSMs, numBlocksPerSm);
	}

	int grid = std::min({
	    minGridSize,                                        // given by cudaOccupancyMaxPotentialBlockSizeVariableSMem
		numSMs * numBlocksPerSm,
	    (int)((array_size + blockSize - 1) / blockSize)   // given by cudaOccupancyMaxActiveBlocksPerMultiprocessor
	});

	return {dim3(grid, 1, 1), dim3(blockSize, 1, 1)};
}
