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
#include "fairlock.hpp"

#include <iostream>
#include <vector>

// Example usage
template <size_t MAX, typename T = int>
struct ProtectedData {
	FairLock fLock;
	T sharedData;

	std::vector<int> ids { std::vector<int>(MAX) }; // TODO
	std::atomic<size_t> it {0};
	size_t count {0};

public:
	void Write(int id) {
		// There is not any guarantee that the threads are actually created with
		// equivalent latency. So they will arrive here in arbitrary order.
		// This code is actually simple.
		// We atomically store the id order of arriving before the lock
		// (unprotected, but atomically). And then we check that the same order
		// is respected in the lock take (inside the lock).
		ids[it++] = id;

		fLock.Lock();
		T readed = sharedData;
		sharedData = id;
		std::cout << "Writer " << id
		          << " readed " << readed
		          << " and wrote : " << sharedData << std::endl;

		if (count > 0)
			myassert(readed == ids[count++]);

		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		fLock.Unlock();
	}

	operator T() const
	{
		return sharedData;
	}
};

int main(int argc, char **argv) {

	argparser::init(argc, argv);
	const size_t nWriters = argparser::cl<size_t>("Writers");

	// The track vector has a limit that needs to be specified and we cannot
	// exceed.
	myassert(nWriters < 100);
	ProtectedData<100> data;

    std::vector<std::thread> writers;

    for (size_t i = 0; i < nWriters; ++i) {
		writers.emplace_back(&ProtectedData<100>::Write, &data, i);
    }

	std::cout << "All threads created" << std::endl;

    for (std::thread& writer : writers)
		writer.join();

	std::cout << "Final value: " << data << std::endl;

    return 0;
}
