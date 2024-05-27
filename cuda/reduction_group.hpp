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

#include <cooperative_groups.h>

#include <cxxabi.h> // for demanger

namespace cg = cooperative_groups;

template <typename T, T (*TOp)(const T&), T (*TBOp)(const T&, const T&)>
__device__ void reduceNGroupInternal(T *data, const size_t size, T* output)
{
    extern __shared__ T sharedData[]; // must have blockdim

	cg::grid_group grid = cg::this_grid();
    cg::thread_block cta = cg::this_thread_block();

	int tid = cta.thread_rank();

    T localValue = 0;

    // Load data into local variable
    for (int idx = grid.thread_rank(); idx < size; idx += grid.size())
        localValue = TBOp(localValue, TOp(data[idx]));

    sharedData[tid] = localValue;
    cta.sync();

    // Perform reduction in shared memory
    for (int stride = cta.size() / 2; stride >= 32; stride >>= 1) {
        if (tid < stride)
            sharedData[tid] = TBOp(sharedData[tid], sharedData[tid + stride]);
        cta.sync();
    }

    if (tid < 32)
    {
        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

		localValue = (tid < size ? sharedData[tid] : 0);

        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            localValue = TBOp(localValue, tile32.shfl_down(localValue, stride));
		}

		// Thread 0 in warp 0
		if (tid == 0) {
			output[cta.group_index().x] = localValue;
		}
    }
}

template <typename T, T (*TOp)(const T&), T (*TBOp)(const T&, const T&)>
__global__ void reduceNGroup(T *data, int size, T* output)
{
	 cg::grid_group grid = cg::this_grid();
	 cg::thread_block cta = cg::this_thread_block();

	 reduceNGroupInternal<T, TOp, TBOp>(data, size, output);
	 grid.sync();

	 const int nBlocks = grid.num_blocks();

	 // The number of blocks should not exceed the 
	 assert(nBlocks <= cta.size());

	 if ((nBlocks > 1) && (cta.group_index().x == 0))
		 reduceNGroupInternal<T, TOp, TBOp>(output, nBlocks, output);
}


template <typename T, typename Op>
typename T::value_type reduceFunGroup(T start, T end, Op fun)
{
	int supportsCoopLaunch = 0;
	if(cudaSuccess != cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, 0) )
		throw std::runtime_error("Cooperative Launch is not supported on this machine.");

	typedef typename T::value_type type;

	int *h_data = &*start;
	int size = std::distance(start, end);

	// Compute max concurrent occupancy combination
	// Compute blockSize; The grid size from here may not be accurate enough because
	// we need to use the sharedSize to get the accurate value for numBlocksPerSm
	int minGridSize = 0, blockSize = 0;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)fun, 0, size);

	const size_t sharedSize = blockSize * sizeof(type);

	int numBlocksPerSm = 0;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, fun, blockSize, sharedSize);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	minGridSize = std::min({
	        deviceProp.multiProcessorCount * numBlocksPerSm,
			blockSize,
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
	cudaMalloc((void**)&d_data, size * sizeof(type));
	cudaMemcpy(d_data, h_data, size * sizeof(type), cudaMemcpyHostToDevice);

	type *d_result;
	cudaMalloc((void**)&d_result, minGridSize * sizeof(type));

	dim3 dim_block(blockSize, 1, 1);
	dim3 dim_grid(minGridSize, 1, 1);
	void* kernelArgs[] = { (void*)&d_data, &size, (void*)&d_result};

	cudaLaunchCooperativeKernel((void *) fun, dim_grid, dim_block, kernelArgs, sharedSize);

	if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
		cudaDeviceSynchronize();
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
	}

	type result;
	cudaMemcpy(&result, d_result, sizeof(type), cudaMemcpyDeviceToHost);

	cudaFree(d_result);
	cudaFree(d_data);

	return result;
}


template <typename T>
typename T::value_type reduceGroup(T start, T end)
{
	using type = typename T::value_type;
	return reduceFunGroup(start, end, reduceNGroup<type, __unit<type>, __sum<type>>);
}
