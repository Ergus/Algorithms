// Copyright (C) 2025  Jimmy Aguilar Mena

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.	 If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include <vector>
#include <cstdint>
#include <optional>
#include <cmath>
#include <functional>
#include <type_traits>
#include <cassert>

#include "utils.h"

template <typename T>
concept Hashable = requires(const T& a) {
	{ std::hash<T>{}(a) } -> std::convertible_to<uint64_t>;
};

template <typename Hasher, typename Key>
concept InvocableHasher = requires(Hasher h, const Key& k) {
	{ h(k) } -> std::convertible_to<uint64_t>;
};

template <
	Hashable key_t,
	typename value_t,
	InvocableHasher<key_t> hasher_t = std::hash<key_t>,
	double max_load_factor = 0.75 >
class SwissTable {

private:

	enum Metadata : uint8_t {
		EMPTY = 0b10000000,	   // 0x80
		DELETED = 0b11111110,  // 0xFE
		SENTINEL = 0b11111111, // 0xFF
	};

	using Entry = std::pair<key_t, value_t>;

	static constexpr hasher_t _hasher {};

	size_t _size {0};

	std::vector<uint8_t> _metadata = std::vector<uint8_t>(16, EMPTY);
	std::vector<Entry> _entries = std::vector<Entry>(16);

	void rehash(size_t new_capacity)
	{
		size_t new_cap = get_pow_2(new_capacity);
		if (new_capacity % 2) {
			new_cap <<= 1;
		}

		SwissTable<key_t, value_t, hasher_t> new_table;
		new_table._metadata.resize(new_cap, EMPTY);
		new_table._entries.resize(new_cap);

		const size_t cap = _entries.size();

		for (size_t i = 0; i < cap; ++i) {
			if (_metadata[i] & EMPTY)
				continue;
			new_table.insert(_entries[i].first, _entries[i].second);
		}

		*this = std::move(new_table);
	}

	// Find the position of the key in the table, or a position to insert it
	std::pair<size_t, bool> find_or_insert_pos(uint64_t hash, const key_t &key) const
	{
		uint64_t cap = capacity();
		assert(cap > 0);

		const size_t pos = hash & (cap - 1);
		const uint8_t h2 = (hash >> 7) & 0x7F;

		for (size_t i = pos; i < _metadata.size(); ++i) {

			// Found empty slot
			if (_metadata[i] == EMPTY)
				return {i, false};

			// Found potential match
			if (_metadata[i] == h2 && _entries[i].first == key)
				return {i, true};
		}

		return {cap, false};
	}

public:

	SwissTable() = default;

	size_t size() const { return _size; }
	size_t capacity() const { return _entries.size(); }

	bool insert(const key_t& key, const value_t& value)
	{
		const size_t cap = capacity();
		if (size() >= cap * max_load_factor) {
			rehash(cap << 1);
		}

		const uint64_t hash = _hasher(key);
		auto [pos, found] = find_or_insert_pos(hash, key);

		if (found) {
			// Update existing key
			_entries[pos].second = value;
			return false;
		}

		// Not found
		assert(pos != cap);
		// Insert new key
		_metadata[pos] = (hash >> 7) & 0x7F;
		_entries[pos] = {key, value};
		++_size;
		return true;
	}

	const std::optional<value_t> find(const key_t& key) const
	{
		assert(_size > 0);

		const uint64_t hash = _hasher(key);
		auto [pos, found] = find_or_insert_pos(hash, key);

		if (found)
			return std::optional<value_t>(_entries[pos].second);

		return std::nullopt;
	}

	bool remove(const key_t& key)
	{
		const uint64_t hash = _hasher(key);
		auto [pos, found] = find_or_insert_pos(hash, key);

		if (!found)
			return false;

		// Mark as deleted
		_metadata[pos] = DELETED;
		--_size;

		return true;
	}

};

