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
#include <stdexcept>

template<int I>
class QueueLock_t {
private:

	class helper {

		friend class QueueLock_t;
		QueueLock_t* _mutex{nullptr};
        std::atomic<helper*> _next{nullptr};
        std::atomic<bool> _status{false};

		void lock(QueueLock_t *m) {
			_mutex = m;
            _next.store(nullptr, std::memory_order_relaxed);
            _status.store(0U, std::memory_order_relaxed);

            // x86 compare exchange operation always has a strong fence
            // "sending" the fields initialized above to other processors.
            if (helper* pred = m->_last.exchange(this); pred != nullptr) {
                if (pred->_next.exchange(this, std::memory_order_release) != nullptr)
					throw std::runtime_error("There was another successor");

				while (!_status)
					std::this_thread::yield();
            }
        }

		void unlock()
        {
			assert(_mutex != nullptr);

			// If there is not _next, then there are two possibilities.
			// 1. This is the last thread in the queue
			// 2. There is a race condition because there is a value in the
			// queue not set in _next YET
            if (_next.load(std::memory_order_relaxed) == nullptr) {

                if (helper* expected = this;
				    _mutex->_last.compare_exchange_strong(expected, nullptr)) {
                    // this was the only item in the queue, and the queue is now empty.
					_mutex = nullptr;
                    return;
                }
                // If we are here is because there is some entry in the queue
                // that is not this (compare_exchange return false)
				// But we need to help because that value is not set (yet) into 
				// _next
				while (_next == nullptr)
					std::this_thread::yield();
            }

            _next.load(std::memory_order_acquire)->_status.store(true, std::memory_order_release);
			_mutex = nullptr;
        }
	};

	// Negative value indicates a write lock, positive indicates the number of read locks.
    std::atomic<helper*> _last{nullptr};

	static thread_local helper h;

public:
	void Lock()
	{
		h.lock(this);
	}

	void Unlock()
	{
		h.unlock();
	}
};

template<int I>
thread_local typename QueueLock_t<I>::helper QueueLock_t<I>::h;

using QueueLock = QueueLock_t<0>;
