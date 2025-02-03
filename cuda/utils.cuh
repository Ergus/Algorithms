// Copyright (C) 2025  Jimmy Aguilar Mena

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

#pragma once

float myGetElapsed(const cudaEvent_t &eStart, const cudaEvent_t &eStop)
{
	cudaEventSynchronize(eStop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, eStart, eStop);
	return milliseconds;
}

class cudaErrorCheck {
public:
	cudaErrorCheck &operator=(cudaError_t err)
	{
		if (err != cudaSuccess) {
			cudaDeviceSynchronize();
			printf("CUDA Error: %s\n", cudaGetErrorString(err));
		}
		return *this;
	}
};


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
	This was expected to be available, but it is not
 **/
template <typename T>
__device__ void swap(T &v1, T &v2)
{
	T tmp = v1;
	v1 = v2;
	v2 = tmp;
}
