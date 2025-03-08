// Copyright (C) 2025  Jimmy Aguilar Mena

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


#include <map>
#include <array>
#include <optional>
#include <stdexcept>

/**
   This is a vector map which stores values in chunks

   While this sounds sumy it actually improves performance because the
   binary tree search O(log(n)) is immediately reduces n by a factor of
   chunk_size.  And within the chunks the access is O(1), but also it
   makes a more efficient use of cache lines and memory access. However,
   if the values are very sparse this wastes a lot of memory as the
   chunks are fully reserved.
 **/
template<typename key_t, typename value_t, size_t chunk_size = 32>
class vectorMap {

	struct storage_t {
		const key_t _start;
		using array_t = std::array<std::optional<value_t>, chunk_size>;

		array_t _values;

		storage_t(key_t start)
			: _start(start)
		{
			assert(_start % chunk_size == 0);
		}

		bool operator<(key_t key) const
		{
			return key < _start;
		}

		bool operator>(key_t key) const
		{
			return key >= (key_t)(_start + chunk_size);
		}

		bool operator==(key_t key) const
		{
			return !(*this < key || *this > key);
		}

		bool operator!=(key_t key) const
		{
			return (*this < key || *this > key);
		}

		std::optional<value_t> &operator[](key_t key)
		{
			if (*this != key)
				throw std::out_of_range("Failed to get bounds");

			std::optional<value_t> &val = _values[key - _start];
			if (!val.has_value())
				val = value_t();

			return val;
		}

		const std::optional<value_t> &at(key_t key) const
		{
			if (*this != key)
				throw std::out_of_range("Failed to get bounds");

			return _values[key - _start];
		}
	};

	std::map<key_t, storage_t> storage;

public:

	vectorMap() = default;

	value_t& operator[](key_t key)
	{
		const key_t start = (key / chunk_size) * chunk_size;

		auto lb_it = storage.lower_bound(start);

		if (lb_it->first != start) {
			lb_it = storage.emplace_hint(lb_it, start, start);
		}

		std::optional<value_t> &val = lb_it->second[key];
		assert(val.has_value());

		return val.value();
	}

	/**
	   Traverse the container chunks with a function

       This is intended for debugging purposes
	 **/
	template <typename F>
	void traverse_chunks(F fun) const
	{
		for (const std::pair<const key_t, storage_t> &entry_pair : storage) {
			fun(entry_pair.first, entry_pair.second._values);
		}
	}


	template <typename F>
	void traverse(F fun) const
	{
		for (const std::pair<const key_t, storage_t> &entry_pair : storage) {
			for (size_t i = 0; i < chunk_size; ++i) {

				key_t key = entry_pair.second._start + i;
				const std::optional<value_t> &entry = entry_pair.second.at(key);

				if (entry.has_value())
					fun(std::pair<key_t, value_t>(key, entry.value()));
			}
		}
	}
};
