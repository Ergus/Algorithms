#pragma once
// Copyright (C) 2024  Jimmy Aguilar Mena

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

template <typename T>
__device__ bool __less(const T& a,const T& b)
{
	return a < b;
}

template <typename T, bool (*TComp)(const T& a,const T& b)>
__global__ void bitonicKernel(T *dev_values, int j, int k)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int ixj = i^j;

	if (ixj <= i)
		return;

	T tmpi = dev_values[i];
	T tmpixj = dev_values[ixj];

	/* The threads with the lowest ids sort the array. */
	if (i&k ? TComp(tmpi, tmpixj) : TComp(tmpixj, tmpi))
	{
		dev_values[i] = tmpixj;
		dev_values[ixj] = tmpi;
	}
}

template <size_t blockdim, typename T, auto TComp>
void bitonicSortBase(T first, T last)
{
	using type = typename T::value_type;

	type *h_data = &*first;
	const size_t size = std::distance(first, last);

	int *d_data;
	cudaMalloc((void**)&d_data, size * sizeof(type));
    cudaMemcpy(d_data, h_data, size * sizeof(type), cudaMemcpyHostToDevice);

	const int nblocks = (size + blockdim - 1) / blockdim;

	for (size_t k = 2; k <= size; k <<= 1)
		for (size_t j = k>>1; j>0; j >>= 1)
			bitonicKernel<type, TComp><<<nblocks, blockdim>>>(d_data, j, k);

	cudaMemcpy(h_data, d_data, size * sizeof(type), cudaMemcpyDeviceToHost);
	cudaFree(d_data);
}


template <size_t blockdim, typename T>
void bitonicSort(T start, T end)
{
	bitonicSortBase<blockdim, T, __less<typename T::value_type>>(start, end);
}

