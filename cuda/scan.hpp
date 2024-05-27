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

/**
   Initial scan that performs the scan within a block.

   In order to perform the scan this kernel reads 2 values/ thread from global
   memory. With offset = blockDim.x and compy then into shared-memory.
   @tparam T underlying type
   @param[in] idata input array
*/
template <typename T>
__global__ void prescan(const T *idata, int size, T *odata, T *tmp = nullptr)
{
	extern __shared__ T temp[];       // allocated on invocation

	int dataSize = 2 * blockDim.x;

	int tid = threadIdx.x;
	int gidx1 = blockIdx.x * dataSize + tid;
	int gidx2 = gidx1 + blockDim.x;

	temp[tid]              = (gidx1 < size ? idata[gidx1] : 0);       // load input into shared memory
	temp[tid + blockDim.x] = (gidx2 < size ? idata[gidx2] : 0);

	__syncthreads();
	const T localLast = temp[2 * blockDim.x - 1];

	const int tid2 = 2 * tid + 1;

	// Up-Sweep
	int offset = 1;
	for (int d = dataSize / 2; d > 0; d /= 2)    // build sum in place up the tree
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
		temp[2 * blockDim.x - 1] = 0;            // clear the last element

	// Down-Sweep
	for (int d = 1; d < dataSize; d *= 2)        // traverse down tree & build scan
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
	if (gidx1 < size)
		odata[gidx1] = temp[tid];

	if (gidx2 < size)
		odata[gidx2] = temp[tid + blockDim.x];	

	if (tmp != 0 && tid == blockDim.x - 1)
		tmp[blockIdx.x] = temp[dataSize - 1] + localLast; // Put the last element as the sum
}

template <typename T>
__global__ void postscan(T *odata, int n, const T *tmp) {

	int tid = threadIdx.x;
	int gidx = blockIdx.x * blockDim.x * 2 + tid;

	const T value = tmp[blockIdx.x];

	// Write output out
	if (gidx < n)
		odata[gidx] += value;

	if (gidx + blockDim.x < n)
		odata[gidx + blockDim.x] += value; 
}

/**
   Perform scan for arrays on devide

   This algorith is divided into tww conditions:
   1. For n < 1024 it basically tries to perform the reduction in a single block
   with a blockdim power of two and ignores the template parameter blockdim.
   2. For n < 1024 it starts the scan in three steps:

   a) perform an inclusive Scan/block and return the result of the scan of every
   individual block + an array of the total sum of elements in every block.

   b) Perform scan_internal ovet rhe array os sums to get the global
   offset that needs to be applied to every block.

   c) Perform a postscan step to update all the individual block indices adding
   the global offsets.
*/
template <int blockdim, typename T>
void scan_internal(T *data, int n, T *o_data)
{
	// Assert blockdim is a power of two;
	static_assert((blockdim & (blockdim - 1)) == 0);

	constexpr size_t blockDataDim = 2 * blockdim;

	// When we can scan in a single pass, then that will be more efficient
	if (n <= 1024) {
		// Compute a blockdim power of two.
		const double half = (double) n / 2.0;
		double exponent = std::ceil(log2(half));
		int bd = std::pow(2, exponent);

		prescan<<<1, bd, 2 * bd * sizeof(T)>>>(data, n, o_data);

	} else {
		const int nblocks = (n + blockDataDim - 1) / blockDataDim;
		T *o_tmp;
		cudaMalloc((void**)&o_tmp, nblocks * sizeof(T));

		const size_t sharedSize = blockDataDim * sizeof(T);

		prescan<<<nblocks, blockdim, sharedSize>>>(data, n, o_data, o_tmp);

		scan_internal<blockdim, T>(o_tmp, nblocks, o_tmp);

		postscan<<<nblocks, blockdim>>>(o_data, n, o_tmp);

		cudaFree(o_tmp);
	}
}

/**
   Cuda scan with similar format than the std::*_scan
   @tparam blockdim the blockdim to use in the device
 */
template <bool EXCLUSIVE, int blockdim, typename Ti, typename To>
void scan(Ti start, Ti end, To oStart)
{
	using type = typename Ti::value_type;

	type *data = &*start;
	const size_t size = std::distance(start, end);

	type *d_data;
	cudaMalloc((void**)&d_data, size * sizeof(type));
    cudaMemcpy(d_data, data, size * sizeof(type), cudaMemcpyHostToDevice);

	// The output data on the device (remove this latter)
	type *o_data;
	cudaMalloc((void**)&o_data, size * sizeof(type));

	scan_internal<blockdim, type>(d_data, size, o_data);

	typename To::value_type *outPtr = &*oStart;
	if (EXCLUSIVE)
		cudaMemcpy(outPtr, o_data, size * sizeof(type), cudaMemcpyDeviceToHost);
	else {
		cudaMemcpy(outPtr, o_data + 1, (size - 1) * sizeof(type), cudaMemcpyDeviceToHost);
		outPtr[size - 1] = outPtr[size - 2] + data[size - 1];
	}

	cudaFree(d_data);
	cudaFree(o_data);
}

template <int blockdim, typename T, typename To>
void exclusive_scan(T start, T end, To oStart)
{
	scan<true, blockdim>(start, end, oStart);
}

template <int blockdim, typename T, typename To>
void inclusive_scan(T start, T end, To oStart)
{
	scan<false, blockdim>(start, end, oStart);
}



