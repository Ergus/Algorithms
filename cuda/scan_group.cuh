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
#include <utility>
#include <algorithm>

#include "utils.cuh"
#include <cxxabi.h>

namespace cg = cooperative_groups;

template <typename T>
__device__ void scan_cta(T *idata, int size)
{
    cg::thread_block cta = cg::this_thread_block();

	assert(size == 2 * cta.size());
	const int tid = cta.thread_rank();
	const int tid2 = 2 * tid + 1;

	// Up-Sweep
	int offset = 1;
	for (int d = size / 2; d > 0; d /= 2)    // build sum in place up the tree
	{
		cta.sync();
		if (tid < d) {
			int ai = offset * tid2 - 1;
			int bi = offset * (tid2 + 1) - 1;
			idata[bi] += idata[ai];
		}
		offset *= 2;
	}

	if (tid == 0)
		idata[2 * cta.size() - 1] = 0;            // clear the last element

	// Down-Sweep
	for (int d = 1; d < size; d *= 2)        // traverse down tree & build scan
	{
		offset /= 2;
		cta.sync();
		if (tid < d) {
			 int ai = offset * tid2 - 1;
			 int bi = offset * (tid2 + 1) - 1;
			 T t = idata[ai];
			 idata[ai] = idata[bi];
			 idata[bi] += t;
		}
	}
}

template <typename T>
__global__ void scanGroupInternal(T *data, const size_t size, T* tmp)
{
	cg::grid_group grid = cg::this_grid();
	cg::thread_block cta = cg::this_thread_block();

	T previous_sum = 0;

	const int tid = cta.thread_rank();
	const int bdim = cta.size();

	const int dataSize = 2 * bdim;
	assert(grid.group_dim().x <= dataSize);  // Required for intermediate reduction.
	extern __shared__ T sharedData[];        // size == datasize == 2 blockDim.x


	if (cta.group_index().x == 0) {
		tmp[tid] = T();
		tmp[tid + bdim] = T();
	}

	// We will iterate now over the data. As every block uses an array
	// of 2 * blockDim.x, then on every iteration we process:
	// 2 * grid.size().
	for (size_t start = 0; start < size; start += (2 * grid.size())) {

		int gidx = start + cta.group_index().x * dataSize + tid;

		// Copy 2 values / thread to shared memory
		sharedData[tid] = (gidx < size) ? data[gidx] : 0;
		sharedData[tid + bdim] = (gidx + bdim < size) ? data[gidx + bdim] : 0;

		// Only need to sync / block (cta)
		cta.sync();

		// get the last value before overwriting it, to sum latter
		const T localLast = sharedData[dataSize - 1];
		const T globalLast = data[start + 2 * grid.size() - 1];

		// Ok, scan in place.. this is the key part
		scan_cta(sharedData, dataSize);

		if (grid.group_dim().x > 1) {
			// And now every block's tid 0 writes the sum of elements
			// (start of next chunk) in global memory
			if (tid == 0)
				tmp[cta.group_index().x] = localLast + sharedData[dataSize - 1];

			if (cta.group_index().x == 0) {
				// I use a portion of the output as a temporal to save
				// the shared data.
				// I need to add start because the previous chunks
				// were already written
				swap(sharedData[tid], tmp[tid]);
				swap(sharedData[tid + bdim], tmp[tid + bdim]);

				cta.sync();

				// scan
				scan_cta(sharedData, dataSize);

				cta.sync();

				// Ok, copy back now
				swap(sharedData[tid], tmp[tid]);
				swap(sharedData[tid + bdim], tmp[tid + bdim]);
			}

			grid.sync();

			previous_sum += tmp[cta.group_index().x];

			// We need to wait here because the next step will override "output"
			grid.sync();
		}

		if (gidx < size)
			data[gidx] = sharedData[tid] + previous_sum;

		if (gidx + bdim < size)
			data[gidx + bdim] = sharedData[tid + bdim] + previous_sum;

		grid.sync();

		// And now get the previous sum for the next iteration
		previous_sum = data[start + 2 * grid.size() - 1] + globalLast;
	}
}

template<typename T>
size_t get_shared_mem(size_t bs)
{
	return 2 * bs * sizeof(T);
}

template <bool EXCLUSIVE, typename Ti>
void scan_group(Ti start, Ti end, Ti o_start) 
{
	// Events to measure time
	cudaEvent_t eStart, eStop;
	cudaEventCreate(&eStart);
	cudaEventCreate(&eStop);

	// Check that cooperative groups are supported
	int supportsCoopLaunch = 0;
	if(cudaSuccess != cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, 0) )
		throw std::runtime_error("Cooperative Launch is not supported on this machine.");

	typedef typename Ti::value_type type;

	const type *h_data = &*start;
	const size_t size = std::distance(start, end);
	type *h_output = &*o_start;
	
	const type last_value = h_data[size - 1];
	// Compute max concurrent occupancy combination
	// Compute blockSize; The grid size from here may not be accurate enough because
	// we need to use the sharedSize to get the accurate value for numBlocksPerSm
	int minGridSize = 0, blockSize = 0;
	cudaError_t err = cudaOccupancyMaxPotentialBlockSizeVariableSMem(
												   &minGridSize,
												   &blockSize,
												   scanGroupInternal<type>,
												   get_shared_mem<type>,
												   size
												  );
	if (err != cudaSuccess) {
		cudaDeviceSynchronize();
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
	}

	blockSize = get_pow_2(blockSize);

	const size_t sharedMem = 2 * blockSize * sizeof(type);

	// Compute how many blocks for fun fit in an SM
	int numBlocksPerSm = 0;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(
												  &numBlocksPerSm,
												  scanGroupInternal<type>,
												  blockSize,
												  sharedMem
												 );

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	// Ok, now compute the grid size.
	// This is needed because to perform grid.sync() all the threads need to be active
	minGridSize = std::min({
	    minGridSize,                                        // given by cudaOccupancyMaxPotentialBlockSizeVariableSMem
	    deviceProp.multiProcessorCount * numBlocksPerSm,    // given by cudaOccupancyMaxActiveBlocksPerMultiprocessor
		(int)(size + (2 * blockSize) - 1) / (2 * blockSize)      // given by input data
	});

	{   // Print the launch information
		const char *fname;
		cudaFuncGetName(&fname, (void *)scanGroupInternal<type>);  // Get name

		int status;
		char *realname = abi::__cxa_demangle(fname, NULL, NULL, &status); // demangle

		fprintf(stderr, "# Launching kernel %s with numBlocks = %d " "blockSize = %d\n",
		        realname, minGridSize, blockSize);

		std::free(realname);
	}

	// Start normal code.
	type *d_data;
	cudaEventRecord(eStart);
	cudaMalloc((void**)&d_data, sharedMem + size * sizeof(type));
	cudaEventRecord(eStop);
	fprintf(stderr, "# cudaMalloc time %f mS\n", myGetElapsed(eStart, eStop));

	cudaEventRecord(eStart);
	cudaMemcpy(d_data, h_data, size * sizeof(type), cudaMemcpyHostToDevice);
	cudaEventRecord(eStop);
	fprintf(stderr, "# cudaMemcpy time %f mS\n", myGetElapsed(eStart, eStop));

	// To make a single cuda malloc
	type *d_result = &d_data[size];

	dim3 dim_block(blockSize, 1, 1);
	dim3 dim_grid(minGridSize, 1, 1);
	void* kernelArgs[] = { (void*)&d_data, (void *)&size, (void*)&d_result};

	cudaEventRecord(eStart);
	err = cudaLaunchCooperativeKernel((void *) scanGroupInternal<type>, dim_grid, dim_block, kernelArgs, sharedMem);
	cudaEventRecord(eStop);

	cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		cudaDeviceSynchronize();
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
	}

	if (EXCLUSIVE)
		cudaMemcpy(h_output, d_data, size * sizeof(type), cudaMemcpyDeviceToHost);
	else {
		cudaMemcpy(h_output, d_data + 1, (size - 1) * sizeof(type), cudaMemcpyDeviceToHost);
		h_output[size - 1] = h_output[size - 2] + last_value;
	}

	cudaFree(d_result);
	cudaFree(d_data);

	fprintf(stderr, "# Kernel time %f mS\n", myGetElapsed(eStart, eStop));
}

template <typename T>
void exclusive_scan_group(T start, T end, T oStart)
{
	scan_group<true>(start, end, oStart);
}

template <typename T>
void inclusive_scan_group(T start, T end, T oStart)
{
	scan_group<false>(start, end, oStart);
}
