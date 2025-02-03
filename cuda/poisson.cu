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

#pragma nv_diag_suppress 815 // suppress consr return warning

#include <vector>
#include <iostream>

#include <argparser.hpp>

#include "utils.h"

#include "poisson.cuh"

template<typename T>
struct BorderMatrixCPU {

public:
	std::vector<T> _data;
	const uint _rows, _cols;

	BorderMatrixCPU(BorderMatrixCPU &) = default;

	/**
	   Matrix with owned storage and border in all dimensions.
	 **/
	BorderMatrixCPU(int rows, int cols)
		: _data((rows + 2) * (cols + 2)), _rows(rows), _cols(cols)
	{
		assert(rows % 32  == 0);
		assert(cols % 32  == 0);
	}

	/**
	   Get a value from the matrix.

       If either i of j is negative it returns a left/up boundary value.
	   Similarly if i == rows or j == _cols this returns the right/down
	   border value
	 */
	T get(int i, int j) const
	{
		return _data[(i + 1) * (_cols + 2) + j + 1];
	}

	/**
	   Get a value from the matrix.

       If either i of j is negative it returns a left/up boundary value.
	   Similarly if i == rows or j == _cols this returns the right/down
	   border value
	 */
	T set(int i, int j, T value)
	{
		assert(i < (int)(_rows + 1));
		assert(j < (int)(_cols + 1));
		return _data[(i + 1) * (_cols + 2) + j + 1] = value;
	}

	T* data()
	{
		return _data.data();
	}

};

template <typename T>
poisson_data<T> poisson_cpu(
	T tolerance, size_t rows, size_t cols,
	BorderMatrixCPU<T> &input, BorderMatrixCPU<T> &output
) {
	poisson_data<T> ret {
		.sum = 0,
		.iterations = 0,
	};

	do {
		ret.sum = 0;

		for (int i = 0; i < input._rows; ++i) {
			for (int j = 0; j < input._cols; ++j) {

				T newval = 0.25 * (input.get(i - 1, j)
								   + input.get(i + 1, j)
								   + input.get(i, j - 1)
								   + input.get(i, j + 1));

				output.set(i, j, newval);

				T diff = (newval - input.get(i, j));
				ret.sum += (diff * diff);
			}
		}

		input._data.swap(output._data);

		ret.iterations += 1;

	} while (ret.sum > tolerance);

	input._data.swap(output._data);

	return ret;
}

int main(int argc, char **argv)
{
	argparser::init(argc, argv);
	const size_t rows = argparser::cl<int>("rows");
	const size_t cols = argparser::cl<int>("cols");
	const size_t printlimit = argparser::cl<int>("print_limit", 512); // Limit to print the vectors

	BorderMatrixCPU<double> imatrix_cpu(rows, cols);

	for (size_t i = 0; i < imatrix_cpu._cols; ++i) {
		imatrix_cpu.set(-1, i, 1);
		imatrix_cpu.set(imatrix_cpu._rows, i, 1);
	}

	for (size_t i = 0; i < imatrix_cpu._rows; ++i) {
		imatrix_cpu.set(i, -1, 1);
		imatrix_cpu.set(i, imatrix_cpu._cols, 1);
	}

	BorderMatrixCPU<double> omatrix_cpu(imatrix_cpu);
	BorderMatrixCPU<double> imatrix_gpu(imatrix_cpu);
	BorderMatrixCPU<double> omatrix_gpu(imatrix_cpu);


	argparser::time t0("poisson_cpu");
	poisson_data<double> data_cpu
		= poisson_cpu(0.1, rows, cols, imatrix_cpu, omatrix_cpu);
	t0.stop();

	std::cout << "CPU Poisson: " << data_cpu.sum << " in:  " << data_cpu.iterations << std::endl;

	argparser::time t1("poisson_gpu");
	poisson_data<double> data_gpu
		= poisson_gpu(0.1, rows, cols, imatrix_gpu.data(), omatrix_gpu.data());
	t1.stop();

	std::cout << "GPU Poisson: " << data_gpu.sum << " in:  " << data_gpu.iterations << std::endl;

	argparser::report<>();

	return 0;
}


