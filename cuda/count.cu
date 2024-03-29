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

#include "utils.h"
#include "count.hpp"

__device__ bool is_five(const int& val)
{
	return val == 5;
}

int main(int argc, char **argv)
{
	argparser::init(argc, argv);
	const size_t size = argparser::cl<int>("array_size");

	std::vector<int> v(size);

	static std::random_device rd;    // you only need to initialize it once
    static std::mt19937 mte(rd());   // this is a relative big object to create

    std::uniform_int_distribution<int> dist(0, 24); // dist(mte)

    std::generate(v.begin(), v.end(), [&dist](){ return dist(mte); });

	// std::cout << v << std::endl;

	std::cout << "C++: " << std::count(v.begin(), v.end(), 5) << std::endl;

	std::cout << "Cuda: " << countWarp<1, int, is_five>(v.begin(), v.end()) << std::endl;



	return 0;
}


