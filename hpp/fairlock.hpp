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
   Fair lock simpler code

 */
class FairLock {
public:
	void Lock() {
		int ticket = _lock.fetch_add(1);

		while (_current != ticket) {
			if (ticket - _current > 1)
				std::this_thread::yield();
		}
	}

	void Unlock() {
		++_current;
	}

private:
	// Negative value indicates a write lock, positive indicates the number of read locks.
	std::atomic<int> _lock{0};
	volatile std::uint16_t _current{0};
};
