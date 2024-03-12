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

#define NUM_BANKS 16
 #define LOG_NUM_BANKS 4

#define CONFLICT_FREE_OFFSET(n)						\
	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 


// int ai = thid;
// int bi = thid + (n/2);
// int bankOffsetA = CONFLICT_FREE_OFFSET(ai)
// int bankOffsetB = CONFLICT_FREE_OFFSET(bi)
// temp[ai + bankOffsetA] = g_idata[ai]
// temp[bi + bankOffsetB] = g_idata[bi] 


template <typename T>
__global__ void prescan(T *idata, int n, T *odata) {

	extern __shared__ T temp[];       // allocated on invocation

	int tid = threadIdx.x;
	int gidx = blockIdx.x * blockDim.x * 2 + tid;
	int offset = 1; 

	temp[tid]     = idata[gidx];       // load input into shared memory
	temp[tid + blockDim.x] = idata[gidx + blockDim.x]; 
		
	const int tid2 = 2 * tid + 1;

	for (int d = n / 2; d > 0; d /= 2)    // build sum in place up the tree
	{
		__syncthreads();
		if (tid < d) { 
			int ai = offset * tid2 - 1;
			int bi = offset * (tid2 + 1) - 1;  
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (tid == 0)
		temp[n - 1] = 0;                  // clear the last element  

	for (int d = 1; d < n; d *= 2)        // traverse down tree & build scan
	{
		offset /= 2;
		__syncthreads();
		if (tid < d) {
			 int ai = offset * tid2 - 1;
			 int bi = offset * (tid2 + 1) - 1; 
			 T t = temp[ai];
			 temp[ai] = temp[bi];
			 temp[bi] += t;
		}
	}
	__syncthreads(); 

	// Write output out
	odata[gidx] = temp[tid];
	odata[gidx + blockDim.x] = temp[tid + blockDim.x]; 
}


template <int blockdim, typename T>
void exclusive_scan(T start, T end)
{
	using type = typename T::value_type;

	size_t size = std::distance(start, end);

	type *h_data = &*start;

	type *d_data;
	cudaMalloc((void**)&d_data, size * sizeof(type));
    cudaMemcpy(d_data, h_data, size * sizeof(type), cudaMemcpyHostToDevice);

	size_t nblocks = (size + blockdim - 1) / blockdim;

	type *o_data;
	cudaMalloc((void**)&o_data, size * sizeof(type));

	const size_t sharedSize = 2 * blockdim * sizeof(type);

	prescan<<<nblocks, blockdim, sharedSize>>>(d_data, size, o_data);

	cudaMemcpy(h_data, o_data, size * sizeof(type), cudaMemcpyDeviceToHost);
}

