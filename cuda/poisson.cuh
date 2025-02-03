#pragma once
/*
 * Copyright (C) 2025  Jimmy Aguilar Mena
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
#include <cooperative_groups/reduce.h>

#include <utils.h>
#include "utils.cuh"

namespace cg = cooperative_groups;

template<typename T>
struct BorderMatrix {

	T *_data;
	uint _rows, _cols;

	/**
	   Matrix with non-owned storage and border in all dimensions.
	 **/
	__device__ BorderMatrix(T *data, int rows, int cols)
		: _data(data), _rows(rows), _cols(cols)
	{
		assert(rows % 32  == 0);
		assert(cols % 32  == 0);
	}

	/**
	   Get a value from the matrix.

       If either i of j is negative it returns a left/up boundary value.
	   Similarly if i == rows or j == _cols this returns the right/down
	   border value
	 */
	__device__ T get(int i, int j) const
	{
		return _data[(i + 1) * (_cols + 2) + j + 1];
	}

	/**
	   Get a value from the matrix.

       If either i of j is negative it returns a left/up boundary value.
	   Similarly if i == rows or j == _cols this returns the right/down
	   border value
	 */
	__device__ T set(int i, int j, T value)
	{
		assert(i < (int)(_rows + 1));
		assert(j < (int)(_cols + 1));

		return _data[(i + 1) * (_cols + 2) + j + 1] = value;
	}

	/**
	   Initialize the matrix
	 **/
	template <typename TGroup>
	__device__ void init(TGroup group, T up, T down, T left, T right, T inside)
	{
		uint fullsize = ((_rows + 1) * (_cols + 1));

		for (uint x = group.thread_rank(); x < fullsize; x += group.num_threads()) {

			int i = (x / (_cols + 2)) - 1;
			int j = (x % (_cols + 2)) - 1;

			T value =
				(i == -1) ? up :
					(i == _rows) ? down :
						(j == -1) ? left :
							(j == _cols) ? right :
								inside;

			// We use absolute indices here.
			set(i, j, value);
		}
	}

	/**
	   Copy the sub-matrix and borders
	 **/
	template <typename TGroup>
	__device__ void copy(
		TGroup &group,
		const BorderMatrix &other,
		uint offset_y, uint offset_x
	) {
		uint fullsize = ((_rows + 2) * (_cols + 2));

		for (uint x = group.thread_rank(); x < fullsize; x += group.num_threads()) {

			// We use absolute indices here.
			uint i = (x / (_cols + 2)) - 1;
			uint j = (x % (_cols + 2)) - 1;

			set(i, j, other.get(offset_y + i, offset_x + j));
		}
	}
};


template <typename T>
__device__ T reduce_block(T value)
{
	cg::thread_block cta = cg::this_thread_block();

	const uint wid = cta.thread_rank() / 32;
	const uint lane = cta.thread_rank() % 32;

	__shared__ T sharedData[32];

	// Now reduce per warp into lanes 0
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1)
		value += __shfl_down_sync(0xffffffff, value, offset);

	// All lanes 0 write the variable to shared memory
	if (lane == 0)
		sharedData[wid] = value;

	__syncthreads();

	// If in warp 0 the perform local reduction
	if (wid == 0) {
		value = (lane < cta.num_threads() / 32) ? sharedData[lane] : 0;

		#pragma unroll
		for (int offset = 16; offset > 0; offset >>= 1)
			value += __shfl_down_sync(0xffffffff, value, offset);
	}

	// Only cta.thread_rank() == 0 returns the right value.
	return value;
}

template <typename T>
struct poisson_data {
	T sum;
	uint iterations;
};

/**
	The idata real dimension is: (dim_y + 2) x (dim_x + 2)
 **/
template <typename T>
__global__ void poisson_solver(
	T *idata, int dim_y, int dim_x,
	T tolerance,
	poisson_data<T> *data
) {
	assert(dim_x % 32  == 0);
	assert(dim_y % 32  == 0);

	// Global data
	cg::grid_group grid = cg::this_grid();
	BorderMatrix<T> imatrix(idata, dim_y, dim_x);

	// Block data (cta)
	cg::thread_block cta = cg::this_thread_block();
	__shared__ T sharedData[34 * 34];
	BorderMatrix<T> shared_matrix(sharedData, 32, 32);

	int counter = 0;

	const int tidx = cta.thread_index().x;
	const int tidy = cta.thread_index().y;

	do {
		T tsum = 0;

		if (grid.thread_rank() == 0)
			data->sum = 0;

		grid.sync();

		for (int i = grid.thread_index().y; i < dim_y; i += grid.dim_threads().y) {
			for (int j = grid.thread_index().x; j < dim_x; j += grid.dim_threads().x) {

				shared_matrix.copy(cta, imatrix, (i / 32) * 32, (j / 32) * 32);
				cta.sync();

				const T oldval = shared_matrix.get(tidy, tidx);

				const T newval = 0.25 * (shared_matrix.get(tidy - 1, tidx)
										 + shared_matrix.get(tidy + 1, tidx)
										 + shared_matrix.get(tidy, tidx - 1)
										 + shared_matrix.get(tidy, tidx + 1));

				// Sum all contributions in this thread
				tsum += ((newval - oldval) * (newval - oldval));

				imatrix.set(i, j, newval);
			}
		}

		// Reduction per cta
		tsum = reduce_block(tsum);

		if (cta.thread_rank() == 0) {
			// Finally add to the global variable
			atomicAdd(&data->sum, tsum);
		}

		++counter;

		grid.sync();

	} while (data->sum > tolerance);

	if (grid.thread_rank() == 0)
		data->iterations = counter;
}

template <typename T>
poisson_data<T> poisson_gpu(T tolerance, size_t rows, size_t cols, const T* input, T* output)
{
	assert(rows % 32 == 0);
	assert(cols % 32 == 0);

	// Events to measure time
	cudaErrorCheck err;
	cudaEvent_t eStart, eStop;
	cudaEventCreate(&eStart);
	cudaEventCreate(&eStop);

	// Check that cooperative groups are supported
	int supportsCoopLaunch = 0;
	if(cudaSuccess != cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, 0) )
		throw std::runtime_error("Cooperative Launch is not supported on this machine.");

	// Compute how many blocks for fun fit in an SM
	int numBlocksPerSm = 0;
	err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&numBlocksPerSm,
		poisson_solver<T>,
		(32 * 32),
		2 * 34 * 34 * sizeof(T)
	);

	cudaDeviceProp deviceProp;
	err = cudaGetDeviceProperties(&deviceProp, 0);

	size_t parallel_blocks = deviceProp.multiProcessorCount;// * numBlocksPerSm;

	std::pair<int, int> dims = closest_divisors(parallel_blocks);

	// Start normal code.
	const size_t full_size = (rows + 2) * (cols + 2);

	T *d_idata;
	cudaEventRecord(eStart);
	cudaMalloc((void**)&d_idata, full_size * sizeof(T) + sizeof(poisson_data<T>));
	cudaEventRecord(eStop);
	fprintf(stderr, "# cudaMalloc time %f mS\n", myGetElapsed(eStart, eStop));
	poisson_data<T> *d_poisson_data = reinterpret_cast<poisson_data<T> *>(&d_idata[full_size]);

	cudaEventRecord(eStart);
	cudaMemcpy(d_idata, input, full_size * sizeof(T), cudaMemcpyHostToDevice);
	cudaEventRecord(eStop);
	fprintf(stderr, "# cudaMemcpy h2d time %f mS\n", myGetElapsed(eStart, eStop));

	dim3 dim_block(32, 32, 1);
	dim3 dim_grid(
		rows < cols ? dims.first : dims.second,
		rows < cols ? dims.second : dims.first,
		1
	);

	void* kernelArgs[] = { (void*)&d_idata, (void*)&rows, (void*)&cols,
						   (void *)&tolerance,
						   (void*)&d_poisson_data};

	cudaEventRecord(eStart);
	err = cudaLaunchCooperativeKernel((void *) poisson_solver<T>, dim_grid, dim_block, kernelArgs, 0);
	cudaEventRecord(eStop);

	cudaDeviceSynchronize();
	fprintf(stderr, "# Kernel time %f mS\n", myGetElapsed(eStart, eStop));

	cudaEventRecord(eStart);
	cudaMemcpy(output, d_idata, full_size * sizeof(T), cudaMemcpyDeviceToHost);
	poisson_data<T> h_poisson_data;
	cudaMemcpy(&h_poisson_data, d_poisson_data, sizeof(poisson_data<T>), cudaMemcpyDeviceToHost);
	cudaEventRecord(eStop);
	fprintf(stderr, "# cudaMemcpy d2h  time %f mS\n", myGetElapsed(eStart, eStop));

	cudaFree(d_idata);

	return h_poisson_data;
}
