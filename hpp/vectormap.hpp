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

template<typename key_t, typename value_t, size_t chunk_size = 10>
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

		bool operator<(key_t key)
		{
			return key < _start;
		}

		bool operator>(key_t key)
		{
			return key >= (key_t)(_start + chunk_size);
		}

		bool operator==(key_t key)
		{
			return !(*this < key || *this > key);
		}

		bool operator!=(key_t key)
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
	};

	std::map<key_t, storage_t> storage;

public:

	vectorMap() = default;

	value_t& operator[](key_t key)
	{
		const key_t start = (key / chunk_size) * chunk_size;

		auto lb_it = storage.lower_bound(start);

		if (lb_it->first != key) {
			lb_it = storage.emplace_hint(lb_it, start, start);
		}

		std::optional<value_t> &val = lb_it->second[key];
		assert(val.has_value());

		return val.value();
	}
};
