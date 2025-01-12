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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.	 If not, see <http://www.gnu.org/licenses/>.
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

	/**
	   The read lock checks the last to bits for cases where there is
       a writer active or pending.
	   Else, it increments the reader lock counter.
	   The result of the increase checks again the result in case of
	   race conditions where one writer increased the lock between our
	   read and the write.
	   We don't use CAS here because the old value is not known in advance.
	 */
	void ReadLock()
	{

		int counter = 0;

		while (true) {
			unsigned int s = _lock.load(std::memory_order_relaxed);
			if (! (s & (WRITER | WRITER_PENDING)) ) { // no writer or pending
				if (_lock.fetch_add(ONE_READER) & WRITER) {
					_lock.fetch_sub(ONE_READER); // some writer got there first, undo the increment
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

	/**
		Simply decreae the counter
	*/
	void ReadUnlock()
	{
		_lock.fetch_sub(ONE_READER);
	}

	/**
	   The write lock is a bit more interesting.
	   As this lock gives some preference to the write operations, but don't
	   enforces any order between the write operations.

	   The steps are simple:
	   1. Read the lock atomic and if not busy (not reader or writer), take it
	   2. Else, if it is busy, then check if there is a pending-writer or
	   attempt to set it myself. This flag enforces that the write operations
	   take precedence over the reads.
	*/
	void WriteLock()
	{
		int counter = 0;

		while (true) {
			unsigned int s = _lock.load(std::memory_order_relaxed);
			if (! (s & BUSY)) {	 // no readers or writer running (so take it)
				if (_lock.compare_exchange_strong(s, WRITER))
					break;           // successfully stored writer flag
			} else if (! (s & WRITER_PENDING) ) {
				_lock.fetch_or(WRITER_PENDING);
			}

			if (counter++ < 16) {
				// _mm_pause(); // Not standard C++... on windows it is, not sure on linux
			} else {
				std::this_thread::yield();
				counter = 0;
			}
		}
	}

	/**
	   The write unlock releases all the read and read pending operations.
	   If some reader tries to take it, it ill check after the set operation
	   that there is not any writer or writer pending.
	   If there is it will politely release the lock with no action
	 */
	void WriteUnlock()
	{
		_lock.fetch_and(READERS);
	}

private:

	static constexpr unsigned int WRITER = 1;                           // 0000000001
	static constexpr unsigned int WRITER_PENDING = 1 << 1;              // 0000000010
	static constexpr unsigned int READERS = ~(WRITER | WRITER_PENDING); // 1111111100
	static constexpr unsigned int ONE_READER = 1 << 2;                  // 0000000100
	static constexpr unsigned int BUSY = WRITER | READERS;              // 1111111101

	std::atomic<unsigned int> _lock{0};
};
