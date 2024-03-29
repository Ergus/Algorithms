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

#include <cstdlib>
#include <array>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>

template <typename T, size_t N = 1024>
class parallelfifolock {
	std::array<T, N> _queue;
	size_t _inIt{0}, _outIt{0};

	std::mutex _lock;

	bool try_push(const T &in)
	{
		std::lock_guard<std::mutex> guard(_lock);

		const size_t tmp = (_inIt + 1) % N;

		if (tmp != _outIt){
			_queue[_inIt] = in;
			_inIt = tmp;
			return true;
		}

		return false;
	}

	bool try_pop(T &out)
	{
		std::lock_guard<std::mutex> guard(_lock);

		if (_outIt != _inIt) {
			out = std::move(_queue[_outIt]);
			_outIt = (_outIt + 1) % N;
			return true;
		}

		return false;
	}

public:

	void push(const T &in)
	{
		while(!try_push(in))
			std::this_thread::yield();
	}


	T pop()
	{
		T tmp;
		while(!try_pop(tmp))
			std::this_thread::yield();

		return tmp;
	}
};
