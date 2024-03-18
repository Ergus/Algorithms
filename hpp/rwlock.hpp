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

        while (true) {
            int expected = 0;
            if ((expected = lock_.load(std::memory_order_acquire)) >= 0
			    && lock_.compare_exchange_weak(expected, expected + 1,
				                               std::memory_order_acquire,
				                               std::memory_order_relaxed))
                break;

            std::this_thread::yield();
        }
    }

    void ReadUnlock() {
		assert(lock_.load() > 0);
        lock_.fetch_sub(1, std::memory_order_release);
    }

    void WriteLock() {
		while (true) {
            int expected = 0;
            if ((expected = lock_.load(std::memory_order_acquire)) == 0
			    && lock_.compare_exchange_weak(expected, -1,
				                               std::memory_order_acquire,
				                               std::memory_order_relaxed))
                break;

            std::this_thread::yield();
        }
    }

    void WriteUnlock() {
		assert(lock_.load() == -1);
        lock_.store(0, std::memory_order_release);
    }

private:
	// Negative value indicates a write lock, positive indicates the number of read locks.
    std::atomic<int> lock_{0};
};
