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
#include <chrono>
#include <vector>

#include "utils.h"

#include "histogram.cuh"

using hrc = std::chrono::high_resolution_clock;

int main(int argc, char **argv)
{
	argparser::init(argc, argv);
	const size_t size = argparser::cl<size_t>("array_size");
	const size_t nbins = argparser::cl<size_t>("nbins");
	const double mean = argparser::cl<double>("mean", 10.0);
	const double stddev = argparser::cl<double>("stddev", 4.0);
	const size_t printlimit = argparser::cl<int>("print_limit", 512); // Limit to print the vectors

	std::vector<double> v(size);

	static std::random_device rd;    // you only need to initialize it once
    static std::mt19937 mte(rd());   // this is a relative big object to create

    std::normal_distribution dist{mean, stddev};
	std::generate(v.begin(), v.end(), [&dist](){ return dist(mte); });

	argparser::time t1("histogram cpu");
	histogram<double, std::vector<unsigned int>> hist_cpu = histogram_cpu(v, nbins);
	t1.stop();

	argparser::time t2("histogram gpu");
	histogram<double, std::vector<unsigned int>> hist_gpu = histogram_gpu(v, nbins);
	t2.stop();

	if (nbins < printlimit) {
		std::cout << hist_cpu << std::endl;
		std::cout << hist_gpu << std::endl;
	}

	assert(hist_cpu == hist_gpu);

	argparser::report<>();

	return 0;
}


