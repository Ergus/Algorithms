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

template <typename T>
constexpr T mysum(const T& lhs, const T& rhs)
{
	return lhs + rhs;
}

/**
   @param[out] output must be numblocks dim
 */
template <typename T>
__global__ void reduceBasicKernel(T *data, const size_t size, T* output)
{
    extern __shared__ T sharedData[]; // must have blockdim

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
	sharedData[tid] = (globalIdx < size ? data[globalIdx] : 0);
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
			sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write the result back to global memory
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}


template <typename T, typename Op>
typename T::value_type reduceFun(T start, T end, Op fun)
{
	typedef typename T::value_type type;

	int *h_data = &*start;
	size_t size = std::distance(start, end);

	type *d_data;
	cudaMalloc((void**)&d_data, size * sizeof(type));
    cudaMemcpy(d_data, h_data, size * sizeof(type), cudaMemcpyHostToDevice);

	const size_t blockdim = 32;
	size_t nblocks = (size + blockdim - 1) / blockdim;

	type *d_result[2];
	cudaMalloc((void**)&d_result[0], nblocks * sizeof(type));

	const size_t sharedSize = blockdim * sizeof(type);

	size_t count = 0;
	fun<<<nblocks, blockdim, sharedSize>>>(d_data, size, d_result[0]);

	if (nblocks > 1)
	{	
		size = nblocks;
		nblocks = (size + blockdim - 1) / blockdim;

		cudaMalloc((void**)&d_result[1], nblocks * sizeof(int));

		while(size > 1)
		{
			fun<<<nblocks, blockdim, sharedSize>>>(
				d_result[count % 2], size, d_result[(count + 1) % 2]
			);

			size = nblocks;
			nblocks = (nblocks + blockdim - 1) / blockdim;
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
	return reduceFun(start, end, reduceBasicKernel<typename T::value_type>);
}
