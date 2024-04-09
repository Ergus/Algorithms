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

#pragma once

#include <atomic>
#include <thread>
#include <cassert>

#include <emmintrin.h> // _mm_pause


/**
   Read-Write lock simpler code

   This lock uses one atomic to declare the lock when the atomic is negative it
   means that there is a running write operation. When it is positive then there
   are some threads reading the variable.

   This lock does not take into account the order of the operations, so the
   write lock cannot be taken until all the read operations finish.
 */
class ReadWriteLock {
public:
	void ReadLock() {
		int counter = 0;
		int expected = 0;
		while (true) {
			if ((expected = _lock.load(std::memory_order_acquire)) >= 0
			    && _lock.compare_exchange_weak(expected, expected + 1,
				                               std::memory_order_acquire,
				                               std::memory_order_relaxed))
				return;

			if (counter++ < 16) {
				// _mm_pause(); // Not standard C++... on windows it is, not sure on linux
			} else {
				std::this_thread::yield();
				counter = 0;
			}
		}
	}

	void ReadUnlock() {
		assert(_lock.load() > 0);
		_lock.fetch_sub(1, std::memory_order_release);
	}

	void WriteLock() {
		int counter = 0;
		while (true) {
			int expected = 0;
			if (_lock.load() == expected // Local Spinning in write lock too
			    && _lock.compare_exchange_weak(expected, -1,
				                               std::memory_order_acquire,
				                               std::memory_order_relaxed))
				return;

			if (counter++ < 16) {
				// _mm_pause(); // Not standard C++... on windows it is, not sure on linux
			} else {
				std::this_thread::yield();
				counter = 0;
			}
		}
	}

	void WriteUnlock() {
		assert(_lock.load() == -1);
		_lock.store(0, std::memory_order_release);
	}

private:
	// Negative value indicates a write lock, positive indicates the number of read locks.
	std::atomic<int> _lock{0};
};
