#pragma once
/*
 * Copyright (C) 2024  Jimmy Aguilar Mena
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <numeric>
#include <vector>
#include <cooperative_groups.h>

#include <cxxabi.h> // for demanger

namespace cg = cooperative_groups;

/**
   Global reduction kernel.

   @tparam T Array type
   @tparam TOp Unary operator to apply to every array element individually
   @tparam TBOp Binary operator to apply to elements pair. Must be associative and commutative
   @param[out] output Must be a global array with size grid.num_blocks()
 **/
template <typename T, T (*TOp)(const T&), T (*TBOp)(const T&, const T&)>
__global__ void reduceGroup(T *data, int size, T* output)
{
	__shared__ T sharedData[32]; // expected to have 32 elements

	cg::grid_group grid = cg::this_grid();
	cg::thread_block cta = cg::this_thread_block();

	cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
	const int lane = warp.thread_rank();
	const int wid = warp.meta_group_rank();

    T localValue = 0;

    // Load data into local variable
    for (int idx = grid.thread_rank(); idx < size; idx += grid.size())
        localValue = TBOp(localValue, TOp(data[idx]));

	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1)
		localValue = TBOp(localValue ,__shfl_down_sync(0xffffffff, localValue, offset));

	if (lane == 0)
		sharedData[wid] = localValue;

	__syncthreads();

	if (wid == 0) {
		localValue = (lane < warp.meta_group_size()) ? sharedData[lane] : 0;

		#pragma unroll
		for (int offset = 16; offset > 0; offset >>= 1)
			localValue = TBOp(localValue ,__shfl_down_sync(0xffffffff, localValue, offset));

		if (lane == 0) {
			output[grid.block_rank()] = localValue;
		}
	}
}


template <typename T, typename Op>
typename T::value_type reduceFunGroup(T start, T end, Op fun)
{
	// Events to measure time
	cudaEvent_t eStart, eStop;
	cudaEventCreate(&eStart);
	cudaEventCreate(&eStop);
	cudaErrorCheck err;

	int supportsCoopLaunch = 0;
	err = cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, 0);

	using type=typename T::value_type;

	int *h_data = &*start;
	int size = std::distance(start, end);

	// Compute max concurrent occupancy combination
	// Compute blockSize; The grid size from here may not be accurate enough because
	// we need to use the sharedSize to get the accurate value for numBlocksPerSm
	int minGridSize = 0, blockSize = 0;
	err = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)fun, 0, size);

	// Compute how many blocks for fun fit in an SM
	int numBlocksPerSm = 0;
	err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, fun, blockSize, 0);

	int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

	std::cout << "Dims: " << minGridSize << " numSMs: " << numSMs << " numBlocksPerSm: " << numBlocksPerSm << std::endl;;

	// For this kernels the max occupancy is the expected behavior.
	// If these values deffer there should be some error, but it I put some printf for testing
	// in the kernel they may defer.
	if (minGridSize == numSMs * numBlocksPerSm) {
		fprintf(stderr, "minGridSize(%d) != numSMs(%d) * numBlocksPerSm(%d)\n",
			minGridSize, numSMs, numBlocksPerSm);
	}

	minGridSize = std::min({
	    minGridSize,
	    numSMs * numBlocksPerSm,
		(size + blockSize - 1) / blockSize
	});

	{   // Print the launch information
		const char *fname;
		cudaFuncGetName(&fname, (void *)fun);  // Get name

		int status;
		char *realname = abi::__cxa_demangle(fname, NULL, NULL, &status); // demangle

		fprintf(stderr, "# Launching kernel %s with numBlocks = %d " "blockSize = %d\n",
		        realname, minGridSize, blockSize);

		std::free(realname);
	}

	// Start normal code.
	type *d_data;
	cudaEventRecord(eStart);
	cudaMalloc((void**)&d_data, size * sizeof(type));
	cudaEventRecord(eStop);
	fprintf(stderr, "# cudaMalloc time %f mS\n", myGetElapsed(eStart, eStop));

	cudaEventRecord(eStart);
	cudaMemcpy(d_data, h_data, size * sizeof(type), cudaMemcpyHostToDevice);
	cudaEventRecord(eStop);
	fprintf(stderr, "# cudaMemcpy time %f mS\n", myGetElapsed(eStart, eStop));

	type *d_result;
	cudaMalloc((void**)&d_result, minGridSize * sizeof(type));

	//dim3 dim_block(blockSize, 1, 1);
	dim3 dim_block(blockSize, 1, 1);
	dim3 dim_grid(minGridSize, 1, 1);
	void* kernelArgs[] = { (void*)&d_data, &size, (void*)&d_result};

	cudaEventRecord(eStart);
	err = cudaLaunchCooperativeKernel((void *) fun, dim_grid, dim_block, kernelArgs);
	cudaEventRecord(eStop);
	fprintf(stderr, "# Kernel time %f mS\n", myGetElapsed(eStart, eStop));

	std::vector<type> result(minGridSize);
	err = cudaMemcpy(result.data(), d_result, minGridSize * sizeof(type), cudaMemcpyDeviceToHost);

	cudaFree(d_result);
	cudaFree(d_data);

	return std::reduce(result.begin(), result.end());
}


template <typename T>
typename T::value_type reduceGroup(T start, T end)
{
	using type = typename T::value_type;
	return reduceFunGroup(start, end, reduceGroup<type, __unit<type>, __sum<type>>);
}
