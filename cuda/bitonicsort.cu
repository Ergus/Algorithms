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
#include <argparser.hpp>
#include <iostream>
#include <random>
#include <algorithm>

#include "utils.h"
#include "bitonicsort.hpp"

// CUDA kernel to perform bitonic merge

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

int main(int argc, char **argv)
{
	argparser::init(argc, argv);
	const size_t size = argparser::cl<int>("array_size");

	std::vector<int> v(size);
	std::iota(v.begin(), v.end(), 1);
	std::shuffle(v.begin(), v.end(), std::mt19937{std::random_device{}()});

	std::cout << v << std::endl;

	std::vector<int> vcopy = v;
	bitonicSort(vcopy.begin(), vcopy.end());
	std::cout << vcopy << std::endl;

	return 0;
}


