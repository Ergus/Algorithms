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

#include <functional>
#include "reduction.cuh"

/**
   Reduction using registers memory
   @tparam T array type
   @tparam N Number of elements to load / thread
   @tparam TOp Binary operator device function defining how to perform the adition.
   default: __device__ T __sum(const T& a,const T& b)
   @param[out] output must be numblocks dim
*/
template <typename T, int N, bool (*TOp)(const T&)>
__global__ void countNWarp(T *data, const size_t size, T* output)
{
	__shared__ T sharedData[32]; // must have blockdim

	const int tid = threadIdx.x;
	const int lane = tid % 32;
	const int wid = tid / 32;

	if (wid == 0)
		sharedData[lane] = 0;

	__syncthreads();

	int globalIdx = blockIdx.x * blockDim.x * N + tid;
	T localValue = 0;

	// Load data into local variable
	for (int i = 0; i < N; ++i)
	{
		bool count = (globalIdx < size) ? TOp(data[globalIdx]) : 0;
		unsigned ballot_result = __ballot_sync(0xffffffff, count);

		if (lane == 0)
			localValue += __popc(ballot_result);

		globalIdx += blockDim.x;
	}

	// All lanes 0 write the variable to shared memory
	if (lane == 0)
		sharedData[wid] = localValue;

	__syncthreads();

	// If in warp 0 the perform local reduction
	if (wid == 0) {
		localValue = sharedData[lane];

		for (int offset = 16; offset > 0; offset >>= 1)
			localValue += __shfl_down_sync(0xffffffff, localValue, offset);
	}

	// Thread 0 in warp 0
	if (tid == 0)
		output[blockIdx.x] = localValue;
}

/**
   Elemental count using warp optimized algorithm and N elements/thread

   This reduction is intended to work beter for very large datasets and a large
   value for N.
   @tparam N Number of elements to load / thread
   @tparam T array type
   @tparam TOp binary operation for count criteria TOp(value) must return true
   for the elements to include in the count.
 */
template <int N, typename T, bool (*TOp)(const T&)>
T countWarp(typename std::vector<T>::iterator start, typename std::vector<T>::iterator end)
{
	return reduceFun2<64, N>(start, end, countNWarp<T, N, TOp>, reduceNWarp<T, N, __unit<T>, __sum<T>>);
}
