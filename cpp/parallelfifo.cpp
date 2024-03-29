/*
 * Copyright (C) 2024  Jimmy Aguilar Mena
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <utils.h>

#include <argparser.hpp>
#include <iostream>

#include "parallelfifolock.hpp"

// Example usage
template <typename T = int>
struct ProtectedData {
	parallelfifolock<T> sharedData;

public:
	void Get(int id, size_t nReads) {
		for (int i = 0; i < nReads; ++i) {
			std::cout << "Reader " << id << " reads shared data: " << sharedData.pop() << std::endl;
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}

	void Set(int id, size_t nWrites) {
		for (int i = 0; i < nWrites; ++i) {
			sharedData.push(id);
			std::cout << "Writer " << id << " writes shared data: " << id << std::endl;
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
		}
	}
};

int main(int argc, char **argv) {

	argparser::init(argc, argv);
	const size_t nWriters = argparser::cl<size_t>("Writers");
	const size_t writes = argparser::cl<size_t>("Writes");

	const size_t nReaders = argparser::cl<size_t>("nReaders");
	const size_t reads = argparser::cl<size_t>("reads");

	const size_t maxIts = std::max(nWriters, nReaders);

	ProtectedData data;

    std::vector<std::thread> readers;
    std::vector<std::thread> writers;

    for (int i = 0; i < maxIts; ++i) {
		if (i < nReaders)
			readers.emplace_back(&ProtectedData<>::Get, &data, i, reads);

		if (i < nWriters)
			writers.emplace_back(&ProtectedData<>::Set, &data, i, writes);
    }

    for (std::thread& reader : readers) {
        reader.join();
    }

    for (std::thread& writer : writers) {
        writer.join();
    }

    return 0;
}
