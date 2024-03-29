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
#include <random>
#include <algorithm>

#include "utils.h"
#include "scan.hpp"

int main(int argc, char **argv)
{
	argparser::init(argc, argv);
	const size_t size = argparser::cl<int>("array_size");
	const size_t printlimit = argparser::cl<int>("print_limit", 512); // Limit to print the vectors

	std::vector<int> v(size);

	static std::random_device rd;    // you only need to initialize it once
    static std::mt19937 mte(rd());   // this is a relative big object to create

    std::uniform_int_distribution<int> dist(0, 20); // dist(mte)

    std::generate(v.begin(), v.end(), [&dist](){ return dist(mte); });

	if (size <= printlimit)
		std::cout << "Initial:\n" << v << std::endl;

	// host scan
	std::vector<int> v1(v.size());
	std::exclusive_scan(v.begin(), v.end(), v1.begin(), 0);

	if (size <= printlimit)
		std::cout << "C++:\n"<< v1 << std::endl;

	// device scan
	std::vector<int> v2(v);
	exclusive_scan<32>(v2.begin(), v2.end());

	if (size <= printlimit)
		std::cout << "Cuda:\n" << v2 << std::endl;

	myassert(v1 == v2);

	return 0;
}


