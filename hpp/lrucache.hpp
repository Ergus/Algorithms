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

#include <unordered_map>
#include <list>
#include <ostream>

template<typename K, typename V, size_t limit>
class lrucache {

	class map_node_t;

	using access_list_t = std::list<typename std::unordered_map<K, map_node_t>::iterator>;

	class map_node_t {
		V value;
		typename access_list_t::iterator access;

		friend class lrucache;

	public:
		explicit map_node_t(V val) : value(val)
		{}

		operator V() const
		{
			return value;
		}

		friend std::ostream& operator<<(std::ostream &out, const map_node_t &node)
		{
			out << node.value;
			return out;
		}

	};

	std::unordered_map<K, map_node_t> map;
	access_list_t access;

	void register_access(typename std::unordered_map<K, map_node_t>::iterator map_node)
	{
		if (map_node->second.access == access.end())
			return;

		access.erase(map_node->second.access);
		map_node->second.access = access.emplace(access.end(), map_node);
	}

public:

	using iterator = typename std::unordered_map<K, map_node_t>::iterator;
	using const_iterator = typename std::unordered_map<K, map_node_t>::const_iterator;

	lrucache()
	{
		map.reserve(limit);
	}

	iterator push(K key, V value)
	{
		iterator it = map.find(key);

		// Insert a new entry when no key existed.
		if (it == map.end()) {
			// If the size is already in the limit, update it.
			if (map.size() == limit)
				erase(access.front()); // remove lru

			assert(map.size() < limit);
			assert(access.size() < limit);
			assert(access.size() == map.size());

			// Ok now insert the new entry and register the insertion as an access.
			auto tmp = map.emplace(key, map_node_t(value));
			assert(tmp.second); // assert that the key was inserted.

			it = tmp.first;

			auto l = it->second.access;

			it->second.access = access.emplace(access.end(), it);
		} else {
			// Update existing entry and register the insertion as an access.
			it->second.value = value;
			register_access(it);
		}

		return it;
	}

	iterator erase(iterator it)
	{
		access.erase(it->second.access);
		return map.erase(it);
	}

	iterator erase(iterator first, iterator last)
	{
		for (iterator it = first; it != last; ++it)
			access.erase(it->access);
		return map.erase(first, last);
	}

	size_t erase(K key)
	{
		iterator it = find(key);

		if (it == map.end())
			return 0;

		erase(it);
		return 1;
	}

	iterator find(K key)
	{
		iterator it = map.find(key);

		if (it == map.end())
			return map.end();

		register_access(it);
		return it;
	}

	V& operator[](K key)
	{
		iterator it = find(key);

		if (it == map.end())
			it = push(key, V());

		return it->second.value;
	}

	size_t size() const
	{
		return map.size();
	}

	size_t max_size() const
	{
		return limit;
	}

	const_iterator begin() const
	{
		return map.begin();
	}

	iterator begin()
	{
		return map.begin();
	}

	const_iterator end() const
	{
		return map.end();
	}

	iterator end()
	{
		return map.end();
	}

	friend std::ostream& operator<<(std::ostream &out, const lrucache &cache)
	{
		out << "[";
		for (auto it : cache.access)
			out << "{" << it->first << ";" << it->second << "} ";
		out << "]";

		return out;
	}

};
