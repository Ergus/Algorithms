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
   Reduction using shared memory
   @tparam T array type
   @tparam N Number of elements to load
   @tparam TOp Unary operator to transform the array input element before adition.
   this function is usefull when instead of reducing we intend to perform other
   operations like counting. default: __device__ T __unit(const T& a) that return a.
   @tparam TBOp Binary operator device function defining how to perform the adition.
   default: __device__ T __sum(const T& a,const T& b) that returns a+b.
   @param[out] output must be numblocks dim
 */
template <typename T, int N, T (*TOp)(const T&), T (*TBOp)(const T&, const T&)>
__global__ void reduceNKernel(T *data, const size_t size, T* output)
{
	extern __shared__ T sharedData[]; // must have blockdim

	int tid = threadIdx.x;

	int globalIdx = blockIdx.x * blockDim.x * N + tid;

	T localValue = 0;

	// Load data into shared memory
	for (int i = 0; i < N && globalIdx < size; ++i)
	{
		localValue = TBOp(localValue, TOp(data[globalIdx]));
		globalIdx += blockDim.x;
	}

	sharedData[tid] = localValue;
	__syncthreads();

	// Perform reduction in shared memory
	#pragma unroll
	for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
		if (tid < stride)
			sharedData[tid] = TBOp(sharedData[tid], sharedData[tid + stride]);
		__syncthreads();
	}

	if (tid < 32)
	{
		volatile T* sdata = sharedData;

		#pragma unroll
		for (int stride = 32; stride > 0; stride >>= 1)
			sdata[tid] = TBOp((T)sdata[tid], (T)sdata[tid + stride]);
	}

	// Write the result back to global memory
	if (tid == 0)
		output[blockIdx.x] = sharedData[0];
}

/**
   Reduction using registers memory
   @tparam T array type
   @tparam N Number of elements to load
   @tparam TOp Binary operator device function defining how to perform the adition.
   default: __device__ T __sum(const T& a,const T& b)
   @param[out] output must be numblocks dim
*/
template <typename T, int N, T (*TOp)(const T&), T (*TBOp)(const T&, const T&)>
__global__ void reduceNWarp(T *data, const size_t size, T* output)
{
	__shared__ T sharedData[32]; // must have blockdim

	const int tid = threadIdx.x;
	const int lane = tid % 32;
	const int wid = tid / 32;

	if (wid == 0)
		sharedData[lane] = 0;

	int globalIdx = blockIdx.x * blockDim.x * N + tid;
	T localValue = 0;

	// Load data into local variable
	for (int i = 0; i < N && globalIdx < size; ++i)
	{
		localValue = TBOp(localValue, TOp(data[globalIdx]));
		globalIdx += blockDim.x;
	}
	__syncthreads();

	// Now reduce per warp into lanes 0
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1)
		localValue = TBOp(localValue, __shfl_down_sync(0xffffffff, localValue, offset));

	// All lanes 0 write the variable to shared memory
	if (lane == 0)
		sharedData[wid] = localValue;

	__syncthreads();

	// If in warp 0 the perform local reduction
	if (wid == 0) {
		localValue = sharedData[lane];

		#pragma unroll
		for (int offset = 16; offset > 0; offset >>= 1)
			localValue = TBOp(localValue, __shfl_down_sync(0xffffffff, localValue, offset));
	}

	// Thread 0 in warp 0
	if (tid == 0)
		output[blockIdx.x] = localValue;
}

/**
   @tparam Tfrac relation of rate data/thread to call. If the kernel accesses 2
   data elements / then this value must be two in order to create less useless
   threads
   @tparam T iterator type
   @tparam Op cuda kernel basic reduction kernel
   @param start interval start iterator
   @param end interval end iterator
   @param fun basic (first) reduction operation
   @param fun2 repeated (second) reduction operation
 */
template <int blockdim, int Tfrac, typename T, typename Op>
typename T::value_type reduceFun2(T start, T end, Op fun, Op fun2)
{
	typedef typename T::value_type type;

	int *h_data = &*start;
	size_t size = std::distance(start, end);

	type *d_data;
	cudaMalloc((void**)&d_data, size * sizeof(type));
	cudaMemcpy(d_data, h_data, size * sizeof(type), cudaMemcpyHostToDevice);

	const size_t step = Tfrac * blockdim;
	size_t nblocks = (size + step - 1) / step;

	type *d_result[2];
	cudaMalloc((void**)&d_result[0], nblocks * sizeof(type));

	const size_t sharedSize = blockdim * sizeof(type);

	size_t count = 0;
	fun<<<nblocks, blockdim, sharedSize>>>(d_data, size, d_result[0]);

	if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
		cudaDeviceSynchronize();
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
	}

	if (nblocks > 1)
	{
		size = nblocks;
		nblocks = (size + step - 1) / step;

		cudaMalloc((void**)&d_result[1], nblocks * sizeof(int));

		while(size > 1)
		{
			fun2<<<nblocks, blockdim, sharedSize>>>(
				d_result[count % 2], size, d_result[(count + 1) % 2]
			);

			if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
				cudaDeviceSynchronize();
				printf("CUDA Error: %s\n", cudaGetErrorString(err));
			}

			size = nblocks;
			nblocks = (nblocks + step - 1) / step;
			++count;
		}
	}

	type result;
	cudaMemcpy(&result, d_result[count % 2], sizeof(type), cudaMemcpyDeviceToHost);

	if (count > 0)
		cudaFree(d_result[1]);

	cudaFree(d_result[0]);

	return result;
}


/**
   Simple wrapper overload for reduceFun2.

   This is only a wrapper around reduceFun2 where fun == fun2. ALl the other
   parameters are the same than reduceFun2.
 */
template <int blockdim, int Tfrac, typename T, typename Op>
typename T::value_type reduceFun(T start, T end, Op fun)
{
	return reduceFun2<blockdim, Tfrac, T, Op>(start, end, fun, fun);
}

/**
   Elemental reduction using basic algorithm and 1 element/thread
 */
template <typename T>
typename T::value_type reduceBasic(T start, T end)
{
	using type = typename T::value_type;
	return reduceFun<64, 1>(start, end, reduceNKernel<type, 1, __unit<type>, __sum<type>>);
}

/**
   Elemental reduction using basic algorithm and N elements/thread
 */
template <int N, typename T>
typename T::value_type reduceN(T start, T end)
{
	using type = typename T::value_type;
	return reduceFun<64, N>(start, end, reduceNKernel<type, N, __unit<type>, __sum<type>>);
}

/**
   Elemental reduction using warp optimized algorithm and N elements/thread
   @tparam N Number of elements to load / thread
   @tparam T iterator type
 */
template <int N, typename T>
typename T::value_type reduceWarp(T start, T end)
{
	using type = typename T::value_type;
	return reduceFun<64, N>(start, end, reduceNWarp<type, N, __unit<type>, __sum<type>>);
}
