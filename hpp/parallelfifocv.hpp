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
#include <mutex>
#include <condition_variable>

template <typename T, size_t N = 1024>
class queue {
	std::array<T, N> _queue;
	size_t _inIt{0}, _outIt{0};

	std::mutex _lock;
	std::condition_variable _cvIn, _cvOut;

public:

	void push(const T &in)
	{
		{
			std::unique_lock lk(_lock);
			while((_inIt + 1) % N == _outIt)
				_cvOut.wait(lk);

			_queue[_inIt] = in;
			_inIt = (_inIt + 1) % N;
		}
		_cvIn.notify_all();
	}

	T pop()
	{
		T res;
		{
			std::unique_lock lk(_lock);
			while(_outIt == _inIt)
				_cvIn.wait(lk);

			res = std::move(_queue[_outIt]);
			_outIt = (_outIt + 1) % N;
		}
		_cvOut.notify_all();
		return res;
	}
};
