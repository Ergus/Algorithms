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

/**
   Simple lru cache using std::ordered_map.

   This lru cache uses an std::ordered_map for constant time access and a double
   linked list to track cache accesses also in constant time.
   The map nodes include an iterator pointing to the node in the double linked
   list while the linked list value itself is an iterator to the map node.
   The cache works by limiting the number of elements the hash table can hold,
   when a new value is inserted after the limit was reached, then the least
   recently access element must be deleted to create space for the new one.
   Tracking the accesses imply that every access need to be tracked in the linked
   list by removing the corresponding node to the end, so the first elements to
   be removed are the corresponding to the ones in the front.
 */
template<typename K, typename V>
class lrucache {

	class map_node_t;

	using access_list_t = std::list<typename std::unordered_map<K, map_node_t>::iterator>;

	/**
	   Wrapper subclass for type

	   We use this because we need to store the value and a pointer (iterator)
	   to the access cache in order to update access register in constant time.
	   This class attempts to behave as similar to V as possible.
	 */
	class map_node_t {
		V value;
		typename access_list_t::iterator access;

		friend class lrucache;

	public:
		explicit map_node_t(V val) : value(val)
		{}

		/** Transparent cast operator */
		operator V() const
		{
			return value;
		}

		/** For easy print overload operator<< */
		friend std::ostream& operator<<(std::ostream &out, const map_node_t &node)
		{
			out << node.value;
			return out;
		}

	};

	size_t _limit;                           // cache elements limit
	std::unordered_map<K, map_node_t> _map;  // map to access throw keys
	access_list_t _access;                   // Maybe we may use mutable here to make other functions const

	void register_access(typename std::unordered_map<K, map_node_t>::iterator map_node)
	{
		if (map_node->second.access == _access.end())
			return;

		_access.erase(map_node->second.access);
		map_node->second.access = _access.emplace(_access.end(), map_node);
	}

public:
	using key_type = typename std::unordered_map<K, map_node_t>::key_type;
	using mapped_type =	V;
	using value_type = std::pair<const K, map_node_t>;

	using iterator = typename std::unordered_map<K, map_node_t>::iterator;
	using const_iterator = typename std::unordered_map<K, map_node_t>::const_iterator;

	lrucache(size_t limit) : _limit(limit)
	{
		_map.reserve(limit);
	}

	iterator push(K key, V value)
	{
		iterator it = _map.find(key);

		// Insert a new entry when no key existed.
		if (it == _map.end()) {
			// If the size is already in the limit, update it.
			if (_map.size() == _limit)
				erase(_access.front()); // remove lru

			assert(_map.size() < _limit);
			assert(_access.size() < _limit);
			assert(_access.size() == _map.size());

			// Ok now insert the new entry and register the insertion as an access.
			auto tmp = _map.emplace(key, map_node_t(value));
			assert(tmp.second); // assert that the key was inserted.

			it = tmp.first;

			it->second.access = _access.emplace(_access.end(), it);
		} else {
			// Update existing entry and register the insertion as an access.
			it->second.value = value;
			register_access(it);
		}

		return it;
	}

	iterator erase(iterator it)
	{
		_access.erase(it->second.access);
		return _map.erase(it);
	}

	iterator erase(iterator first, iterator last)
	{
		for (iterator it = first; it != last; ++it)
			_access.erase(it->access);
		return _map.erase(first, last);
	}

	size_t erase(K key)
	{
		iterator it = find(key);

		if (it == _map.end())
			return 0;

		erase(it);
		return 1;
	}

	iterator find(K key)
	{
		iterator it = _map.find(key);

		if (it == _map.end())
			return _map.end();

		register_access(it);
		return it;
	}

	V& operator[](K key)
	{
		iterator it = find(key);

		if (it == _map.end())
			it = push(key, V());

		return it->second.value;
	}

	size_t size() const
	{
		return _map.size();
	}

	size_t max_size() const
	{
		return _limit;
	}

	const_iterator begin() const
	{
		return _map.begin();
	}

	iterator begin()
	{
		return _map.begin();
	}

	const_iterator end() const
	{
		return _map.end();
	}

	iterator end()
	{
		return _map.end();
	}

	/**
	   Overload operator<< to print the elements in access order

	   This prints from least recent to more recent which is very useful to
	   debug and makes some sense considering that unordered maps have no
	   defined order.
	 */
	friend std::ostream& operator<<(std::ostream &out, const lrucache &cache)
	{
		out << "[";
		for (auto it : cache._access)
			out << "{" << it->first << ";" << it->second << "} ";
		out << "]";

		return out;
	}

};
