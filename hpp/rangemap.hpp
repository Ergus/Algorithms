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

#include <initializer_list>
#include <map>
#include <stdexcept>
#include <limits>
#include "utils.h"
#include <iostream>

template<typename key_t, typename value_t>
class rangeMap {
public:

	using const_iterator = std::map<key_t, value_t>::const_iterator;
	using iterator = std::map<key_t, value_t>::iterator;
	using value_type = std::map<key_t, value_t>::value_type;
	using range = std::pair<iterator,iterator>;

	rangeMap(value_t val = 0)
		: _map({{std::numeric_limits<key_t>::lowest(), val}})
	{
	}

	iterator lower_bound(key_t key)
	{
		const_iterator it = _map.lower_bound(key);

		if (it == _map.begin() || it == _map.end() || it.first == key)
			return it;

		return it.prev();
	}

	const_iterator begin() const { return _map.begin(); }
	const_iterator end() const { return _map.end(); }

	void setRange(key_t start, key_t end, value_t value)
	{
		std::cout << "Set range: [" << start << "; " << end << "] = " << value << std::endl;

		// Inverse range is invalid, empty do nothing.
		if (end <= start)
			return;

		// I call lower_bound again because map has missing hint find. 
		iterator it1 = _map.lower_bound(start);
		// If there were some _map.lower_bound with hint this may be the best
		// place to use it.
		iterator it2 = _map.lower_bound(end);

		// There are 3 cases here
		if (it2 == _map.end()
		    || it2->first > end && std::prev(it2)->second != value) {
			// End key does not exist, but the prev merge
			std::advance(it2, -1);           // step back
			it2 =_map.emplace_hint(it2, end, it2->second);
		} else if (it2->first == end && it2->second == value) {
			// == (key and value) advance to merge
			std::advance(it2, 1);
		}

		// There are 3 cases here
		if (it1 != _map.begin() && std::prev(it1)->second == value) {
			// Start will start in a range with same value (merge)
			std::advance(it1, -1);
		} else if (it1->first == start) {
			// Another range starts in same place, just update value
			it1->second = value;
		} else {
			// OK insert a new entry, hint is prev
			it1 =_map.emplace_hint(std::prev(it1), start, value);
		}

		// Call a single grouped erase due to internal re-balance.
		std::advance(it1, 1);
		_map.erase(it1, it2);
	}

	friend std::ostream& operator<<(std::ostream &out, const rangeMap &map)
	{
		out << map._map;
		return out;
	}

	bool isEqual(std::initializer_list<value_type> list) const
	{
		return _map == std::map<key_t, value_t>(list);
	}

private:
	std::map<key_t, value_t> _map;
};


