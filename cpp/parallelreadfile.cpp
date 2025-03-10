// Copyright (C) 2025  Jimmy Aguilar Mena

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.	 If not, see <http://www.gnu.org/licenses/>.

#include <utils.h>
#include <argparser.hpp>
#include <thread>
#include <iostream>
#include <mutex>

#include "parallelreadfile.hpp"

// Global mutex to stdout...
std::mutex output_mutex;

void my_fun(size_t tid, size_t local_count, std::string &line)
{
	std::lock_guard<std::mutex> lock(output_mutex);
	std::cout << tid << ": "<< local_count << " " << line << std::endl;
}

int main(int argc, char **argv)
{
	argparser::init(argc, argv);

	const std::string filename = argparser::cl<std::string>("filename");
	const size_t num_threads = argparser::cl<size_t>(
		"num_threads",
		std::thread::hardware_concurrency()
	);

	std::cout << "Processing file: " << filename << " with " << num_threads << " threads" << std::endl;

	std::cout << "Loop parallel" << std::endl;
	try {
		my::process_file<my_fun>(filename, num_threads);
	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	std::cout << "Loop recursive" << std::endl;
	try {
		my::process_file2<my_fun>(filename, num_threads);
	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
