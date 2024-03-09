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
__device__ T __sum(const T& a,const T& b)
{
	return a + b;
}

/**
   Reduction using shared memory
   @tparam T array type
   @tparam N Number of elements to load
   @tparam TOp Binary operator device function defining how to perform the adition.
   default: __device__ T __sum(const T& a,const T& b)
   @param[out] output must be numblocks dim
 */
template <typename T, int N, T (*TOp)(const T&, const T&) =  __sum<T>>
__global__ void reduceNKernel(T *data, const size_t size, T* output)
{
    extern __shared__ T sharedData[]; // must have blockdim

    int tid = threadIdx.x;

	int globalIdx = blockIdx.x * blockDim.x * N + tid;

	T localData = 0;

    // Load data into shared memory
	for (int i = 0; i < N && globalIdx < size; ++i)
	{
		localData += data[globalIdx];
		globalIdx += blockDim.x;
	}

	sharedData[tid] = localData;
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride)
			sharedData[tid] = TOp(sharedData[tid], sharedData[tid + stride]);
        __syncthreads();
    }

	if (tid < 32)
	{
		volatile T* sdata = sharedData;
		sdata[tid] = TOp((T)sdata[tid], (T)sdata[tid + 32]);
		sdata[tid] = TOp((T)sdata[tid], (T)sdata[tid + 16]);
		sdata[tid] = TOp((T)sdata[tid], (T)sdata[tid + 8]);
		sdata[tid] = TOp((T)sdata[tid], (T)sdata[tid + 4]);
		sdata[tid] = TOp((T)sdata[tid], (T)sdata[tid + 2]);
		sdata[tid] = TOp((T)sdata[tid], (T)sdata[tid + 1]);
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
template <typename T, int N, T (*TOp)(const T&, const T&) =  __sum<T>>
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
		localValue += data[globalIdx];
		globalIdx += blockDim.x;
	}
	__syncthreads();
	
	// Now reduce per warp into lanes 0
	for (int offset = 16; offset > 0; offset >>= 1)
		localValue = TOp(localValue, __shfl_down_sync(0xffffffff, localValue, offset));

	// All lanes 0 write the variable to shared memory
	if (lane == 0)
		sharedData[wid] = localValue;

	__syncthreads();

	// If in warp 0 the perform local reduction
	if (wid == 0) {
		localValue = sharedData[lane];

		for (int offset = 16; offset > 0; offset >>= 1)
			localValue = TOp(localValue, __shfl_down_sync(0xffffffff, localValue, offset));
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
 */
template <int blockdim, int Tfrac, typename T, typename Op>
typename T::value_type reduceFun(T start, T end, Op fun)
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

	if (nblocks > 1)
	{	
		size = nblocks;
		nblocks = (size + step - 1) / step;

		cudaMalloc((void**)&d_result[1], nblocks * sizeof(int));

		while(size > 1)
		{
			fun<<<nblocks, blockdim, sharedSize>>>(
				d_result[count % 2], size, d_result[(count + 1) % 2]
			);

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

template <typename T>
typename T::value_type reduceBasic(T start, T end)
{
	return reduceFun<64, 1>(start, end, reduceNKernel<typename T::value_type, 1, __sum>);
}

template <int N, typename T>
typename T::value_type reduceN(T start, T end)
{
	return reduceFun<64, N>(start, end, reduceNKernel<typename T::value_type, N>);
}

template <int N, typename T>
typename T::value_type reduceWarp(T start, T end)
{
	return reduceFun<64, N>(start, end, reduceNWarp<typename T::value_type, N>);
}
