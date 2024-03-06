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

#include <random>
#include <algorithm>

template <typename T>
__global__ void bitonicKernel(T *dev_values, int j, int k)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int ixj = i^j;
	if (ixj <= i)
		return;

	T tmpi = dev_values[i];
	T tmpixj = dev_values[ixj];

	/* The threads with the lowest ids sort the array. */
	if (i&k ? tmpi < tmpixj : tmpi > tmpixj)
	{
		dev_values[i] = tmpixj;
		dev_values[ixj] = tmpi;
	}
}

template <typename T>
void bitonicSort(T first, T last)
{
	int *h_data = &*first;
	const size_t size = std::distance(first, last);

	int *d_data;
	cudaMalloc((void**)&d_data, size * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

	const int blockdim = 32;
	const int nblocks = (size + blockdim - 1) / blockdim;

	for (size_t k = 2; k <= size; k <<= 1) {
		for (size_t j = k>>1; j>0; j >>= 1) {
			bitonicKernel<<<nblocks, blockdim>>>(d_data, j, k);
		}
	}

	cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_data);
}


