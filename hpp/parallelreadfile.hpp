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


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <future>
#include <cassert>
#include <numeric>

namespace my {

	template<void (*TOp)(size_t, size_t, std::string&)>
	size_t process_chunk(
		const std::string &filename,
		size_t start, size_t end,
		size_t tid
	) {
		std::ifstream file(filename, std::ios::in);
		if (!file.is_open()) {
			std::cerr << "Failed to open file!" << std::endl;
			return 0;
		}

		// Position the file pointer at the start position
		file.seekg(start);

		// If not at the beginning of the file, find the next newline
		if (start > 0)
			file.ignore(256, '\n');

		std::string line;
		size_t local_count = 0;

		// Read lines until we reach the end position or end of file
		while (file.good() && file.tellg() < (int)end) {
			if (!std::getline(file, line)) {
				break;
			}

			// Parse the line to get name and number
			TOp(tid, local_count, line);

			// Increment counters
			++local_count;
		}

		file.close();
		return local_count;
	}

	template<void (*TOp)(size_t, size_t, std::string&)>
	void process_file(const std::string &filename, size_t num_threads)
	{
		auto filesize = std::filesystem::file_size(filename);
		if (filesize == 0) {
			std::cout << "File is empty." << std::endl;
			return;
		}

		size_t chunkSize = filesize / num_threads;
		std::vector<std::future<size_t>> futures;

		for (size_t i = 0; i < num_threads; ++i) {
			const size_t start_pos = std::min(filesize, i * chunkSize);
			const size_t end_pos = std::min(filesize, (i + 1) * chunkSize);

			if (start_pos == end_pos) {
				assert(start_pos == filesize);
				assert(end_pos == filesize);
				break;
			}

			assert(start_pos < end_pos);

			futures.push_back(
				std::async(
					std::launch::async,
					&process_chunk<TOp>, filename, start_pos, end_pos, i
				)
			);
		}

		// Wait for all threads to complete and get their results
		size_t total_processed = transform_reduce(
			futures.begin(), futures.end(),
			0,
			std::plus{},
			[](auto &future) { return  future.get(); });

		std::cout << "Total records processed: " << total_processed << std::endl;
	}


	template<void (*TOp)(size_t, size_t, std::string&)>
	size_t process_file_rec(
		const std::string &filename,
		size_t start, size_t end,
		size_t num_threads
	) {
		assert(num_threads > 1);

		if (num_threads == 1) {
			return process_chunk<TOp>(filename, start, end, 0);
		}

		size_t half = (start + end) / 2;
		auto dv = std::div(num_threads, 2);

		if (num_threads == 2) {

			std::future<size_t> future = std::async(
				std::launch::async,
				&process_chunk<TOp>, filename, start, half, 0
			);
			size_t local = process_chunk<TOp>(filename, half, end, 0);

			return local + future.get();
		}

		std::future<size_t> future1 = std::async(
			std::launch::async,
			process_file_rec<TOp>, filename, start, half, dv.quot + dv.rem
		);

		std::future<size_t> future2 = std::async(
			std::launch::async,
			process_file_rec<TOp>, filename, half, end, dv.quot
		);

		return future1.get() + future2.get();
	}


	template<void (*TOp)(size_t, size_t, std::string&)>
	void process_file2(const std::string &filename, size_t num_threads)
	{
		auto filesize = std::filesystem::file_size(filename);
		if (filesize == 0) {
			std::cout << "File is empty." << std::endl;
			return;
		}

		size_t total_processed = process_file_rec<TOp>(filename, 0, filesize, num_threads);

		std::cout << "Total records processed: " << total_processed << std::endl;
	}

}
