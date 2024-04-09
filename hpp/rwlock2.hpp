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
   Read-Write lock with write priority

   Unlike the other ReadWriteLock this one reserves one bit to know if there is
   any thread trying to take the lock for writing operations.  In that case the
   write operations take precedence over the read ones. So the read lock are
   only taken when there is no any pending write.

   When some write operation arrives and there are reads taking place, the lock
   is marked to avoid that latter read operations delay the write ones.
 */
class ReadWriteLock2 {
public:
	void ReadLock() {

		int counter = 0;

		while (true) {
            unsigned int s = _lock.load(std::memory_order_relaxed);
            if (! (s & (WRITER | WRITER_PENDING)) ) { // no writer or pending
                if (_lock.fetch_add(ONE_READER) & WRITER) {
					_lock -= ONE_READER; // some writer got there first, undo the increment
                } else {
					break; // successfully stored increased number of readers
				}
            }

			if (counter++ < 16) {
				// _mm_pause(); // Not standard C++... on windows it is, not sure on linux
			} else {
				std::this_thread::yield();
				counter = 0;
			}
        }
	}

	void ReadUnlock() {
        _lock -= ONE_READER;
    }

	void WriteLock() {
		int counter = 0;

		while (true) {
			unsigned int s = _lock.load(std::memory_order_relaxed);
			if (! (s & BUSY)) {  // no readers or writer running (so take it)
				if (_lock.compare_exchange_strong(s, WRITER))
                    break; // successfully stored writer flag
			} else if (! (s & WRITER_PENDING) ) {
				_lock |= WRITER_PENDING;
			}

			if (counter++ < 16) {
				// _mm_pause(); // Not standard C++... on windows it is, not sure on linux
			} else {
				std::this_thread::yield();
				counter = 0;
			}
		}
	}

	void WriteUnlock() {
		_lock &= READERS;
	}

private:

    static constexpr unsigned int WRITER = 1;
    static constexpr unsigned int WRITER_PENDING = 1 << 1;
    static constexpr unsigned int READERS = ~(WRITER | WRITER_PENDING);
    static constexpr unsigned int ONE_READER = 4;
    static constexpr unsigned int BUSY = WRITER | READERS;

	std::atomic<unsigned int> _lock{0};
};
