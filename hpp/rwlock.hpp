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

class ReadWriteLock {
public:
	void ReadLock() {
		int expected = 0;
		while (true) {
			if ((expected = _lock.load(std::memory_order_acquire)) >= 0
			    && _lock.compare_exchange_weak(expected, expected + 1,
				                               std::memory_order_acquire,
				                               std::memory_order_relaxed))
				break;

			std::this_thread::yield();
		}
	}

	void ReadUnlock() {
		assert(_lock.load() > 0);
		_lock.fetch_sub(1, std::memory_order_release);
	}

	void WriteLock() {
		while (true) {
			int expected = 0;
			if (_lock.load() == expected // Local Spinning in write lock too
			    && _lock.compare_exchange_weak(expected, -1,
				                               std::memory_order_acquire,
				                               std::memory_order_relaxed))
				break;

			std::this_thread::yield();
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
