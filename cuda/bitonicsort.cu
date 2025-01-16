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
#include <algorithm>
#include <numeric>
#include <random>

#include "utils.h"
#include "bitonicsort.cuh"

int main(int argc, char **argv)
{
	argparser::init(argc, argv);
	const size_t size = argparser::cl<int>("array_size");
	const size_t printlimit = argparser::cl<int>("print_limit",512); // Limit to print the vectors

	std::vector<int> v(size);
	std::iota(v.begin(), v.end(), 1);
	std::shuffle(v.begin(), v.end(), std::mt19937{std::random_device{}()});

	if (size <= printlimit)
		std::cout << v << std::endl;

	std::vector<int> v1 = v;
	std::sort(v1.begin(), v1.end());

	if (size <= printlimit)
		std::cout << v1 << std::endl;

	std::vector<int> v2 = v;
	bitonicSort<32>(v2.begin(), v2.end());

	if (size <= printlimit)
		std::cout << v2 << std::endl;
	assert(std::is_sorted(v2.begin(), v2.end()));
	myassert(v1 == v2);

	v2 = v;
	bitonicSort<128>(v2.begin(), v2.end());

	if (size <= printlimit)
		std::cout << v2 << std::endl;
	assert(std::is_sorted(v2.begin(), v2.end()));
	myassert(v1 == v2);

	return 0;
}


