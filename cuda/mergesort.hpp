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

#include "bitonicsort.hpp" // for __less

template<auto TComp, typename T>
__global__ void merge(T *d_in, T *d_out, size_t size, size_t width)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t left = idx * 2 * width;
    const size_t right = min(left + width, size);
    const size_t end = min(left + 2 * width, size);

    size_t i = left, j = right;
    size_t k = left;

    while (i < right && j < end)
		d_out[k++] = TComp(d_in[j], d_in[i]) ? d_in[j++] : d_in[i++];

    while (i < right)
        d_out[k++] = d_in[i++];

    while (j < end)
        d_out[k++] = d_in[j++];
}

template<int blockdim, typename T, auto TComp>
void mergeSortBase(T first, T last)
{
	using type = typename T::value_type;

	type *h_data = &*first;
	const size_t size = std::distance(first, last);

    type *d_in, *d_out;
    const size_t bytes = size * sizeof(type);

    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_data, bytes, cudaMemcpyHostToDevice);

    int numBlocks = (size / 2 + blockdim - 1) / blockdim;

    for (size_t width = 1; width < size; width <<= 1) {
        merge<TComp><<<numBlocks, blockdim>>>(d_in, d_out, size, width);

		if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
			cudaDeviceSynchronize();
			printf("CUDA Error: %s\n", cudaGetErrorString(err));
		}

        std::swap(d_in, d_out);
    }

    cudaMemcpy(h_data, d_in, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}


template <size_t blockdim, typename T>
void mergeSort(T start, T end)
{
	mergeSortBase<blockdim, T, __less<typename T::value_type>>(start, end);
}
