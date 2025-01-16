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

#include <argparser.hpp>
#include <iostream>
#include <numeric>
#include <random>
#include <algorithm>
#include <chrono>

#include "utils.h"
#include "reduction.cuh"
#include "reduction_group.cuh"

using hrc = std::chrono::high_resolution_clock;

int main(int argc, char **argv)
{
	argparser::init(argc, argv);
	const size_t size = argparser::cl<int>("array_size");

	std::vector<int> v(size);

	static std::random_device rd;    // you only need to initialize it once
    static std::mt19937 mte(rd());   // this is a relative big object to create

    std::uniform_int_distribution<int> dist(0, 1024); // dist(mte)
	std::generate(v.begin(), v.end(), [&dist](){ return dist(mte); });

	hrc::time_point start = std::chrono::high_resolution_clock::now();
	int cpp = std::reduce(v.begin(), v.end());
	hrc::time_point stop = std::chrono::high_resolution_clock::now();
	auto elaps = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	std::cerr << "# C++ time: " << elaps << std::endl;
	std::cout << "C++: " << cpp << std::endl;

	int basic = reduceBasic(v.begin(), v.end());
	std::cout << "CudaBasic: " << basic << std::endl;
	myassert(basic == cpp);

	int cuda1 = reduceN<1>(v.begin(), v.end());
	std::cout << "Cuda1: " << cuda1 << std::endl;
	myassert(cuda1 == basic)

	int cuda2 = reduceN<2>(v.begin(), v.end());
	std::cout << "Cuda2: " << cuda2 << std::endl;
	myassert(cuda2 == basic);

	int cuda3 = reduceN<3>(v.begin(), v.end());
	std::cout << "Cuda3: " << cuda3 << std::endl;
	myassert(cuda3 == basic);

	int cuda4 = reduceN<4>(v.begin(), v.end());
	std::cout << "Cuda4: " << cuda4 << std::endl;
	myassert(cuda4 == basic);

	int warp1 = reduceWarp<1>(v.begin(), v.end());
	std::cout << "Cuda1Warp: " << warp1 << std::endl;
	myassert(warp1 == basic);

	int warp2 = reduceWarp<2>(v.begin(), v.end());
	std::cout << "Cuda2Warp: " << warp2 << std::endl;
	myassert(warp2 == basic);

	int warp3 = reduceWarp<3>(v.begin(), v.end());
	std::cout << "Cuda3Warp: " << warp3 << std::endl;
	myassert(warp3 == basic);

	int warp4 = reduceWarp<4>(v.begin(), v.end());
	std::cout << "Cuda4Warp: " << warp4 << std::endl;
	myassert(warp4 == basic);

	int group = reduceGroup(v.begin(), v.end());
	std::cout << "Cuda5Group: " << group << std::endl;
	myassert(group == basic);

	return 0;
}


