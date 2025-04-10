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

#pragma once

#include <vector>
#include <cstdint>
#include <optional>
#include <cmath>
#include <functional>
#include <type_traits>

template <typename T>
concept Hashable = requires(const T& a) {
    { std::hash<T>{}(a) } -> std::convertible_to<std::size_t>;
};



// template<>
// struct std::hash<std::string>
// {
//     std::size_t operator()(const std::string& s) const noexcept
//     {
// 		uint64_t hash = 5381;
// 		for (const char &c : s)
// 			hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

// 		return hash;
//     }
// };



template <
	Hashable key_t,
	typename value_t,
	class hasher_t = std::hash<key_t>,
	double max_load_factor = 0.75
	>
class SwissTable {

	//static_assert(std::is_invocable_r_v<std::size_t, std::hash<key_t>(key_t), key_t>);

private:

	enum Metadata : uint8_t {
		EMPTY = 0b00000000,
		FULL = 0b10000000,
		DELETED = 0b01000000,
		HASH_MASK = 0b00111111 // Lower 6 bits of the hash for grouping
	};

	struct Entry {
		uint64_t hash;
		key_t key;
		value_t value;

		Entry() = default;

		Entry(uint64_t h, key_t k, value_t v)
			: hash(h), key(k), value(v)
		{}
	};

	hasher_t hasher;

	static constexpr size_t INITIAL_CAPACITY = 16;
	static constexpr size_t GROUP_SIZE = 16;

    size_t _capacity {INITIAL_CAPACITY};
    size_t _size {0};
    std::vector<uint8_t> _metadata = std::vector<uint8_t>(INITIAL_CAPACITY);
    std::vector<Entry> _buckets = std::vector<Entry>(INITIAL_CAPACITY);

    bool resize(size_t new_capacity)
	{
        std::vector<Entry> old_buckets = std::move(_buckets);
        std::vector<uint8_t> old_metadata = std::move(_metadata);
        size_t old_capacity = _capacity;

        _capacity = new_capacity;
        _buckets.resize(_capacity);
        _metadata.assign(_capacity, EMPTY);
        _size = 0;

        for (size_t i = 0; i < old_capacity; ++i) {
            if (old_metadata[i] & FULL) {
                insert(old_buckets[i].key, old_buckets[i].value);
            }
        }
        return true;
    }

public:

	SwissTable() = default;

    size_t size() const { return _size; }
    size_t capacity() const { return _capacity; }

    bool insert(const key_t& key, value_t value)
	{
        if (static_cast<double>(_size) / _capacity >= max_load_factor) {
            if (!resize(_capacity * 2)) {
                return false;
            }
        }

        const uint64_t hash = hasher(key);
        size_t index = hash % _capacity;
        uint8_t group_hash = static_cast<uint8_t>(hash & HASH_MASK);

        // Probe for an empty or deleted slot within a small group
        for (size_t offset = 0; offset < GROUP_SIZE; ++offset) {
            size_t probe_index = (index + offset) % _capacity;
            if (!(_metadata[probe_index] & FULL)) { // Empty or Deleted
                _buckets[probe_index] = Entry(hash, key, value);
                _metadata[probe_index] = FULL | group_hash;
                ++_size;
                return true;
            }
        }

        // If no space in the initial group, perform linear probing
        size_t current_index = index;
        while (_metadata[current_index] & FULL) {
            current_index = (current_index + 1) % _capacity;
        }

        _buckets[current_index] = Entry(hash, key, value);
        _metadata[current_index] = FULL | group_hash;
        ++_size;
        return true;
    }

    const std::optional<value_t> find(const key_t& key) const
	{
        uint64_t hash = hasher(key);
        size_t index = hash % _capacity;
        uint8_t group_hash = static_cast<uint8_t>(hash & HASH_MASK);

        // First, check the "group" of GROUP_SIZE around the initial index
        for (size_t offset = 0; offset < GROUP_SIZE; ++offset) {
            size_t probe_index = (index + offset) % _capacity;
            if ((_metadata[probe_index] & FULL) &&
                (_metadata[probe_index] & HASH_MASK) == group_hash &&
                _buckets[probe_index].hash == hash &&
                _buckets[probe_index].key == key) {
                return std::optional(_buckets[probe_index].value);
            }
        }

        // If not found in the group, perform linear probing
        size_t current_index = index;
        while (_metadata[current_index] & FULL) {
            if ((_metadata[current_index] & HASH_MASK) == group_hash &&
                _buckets[current_index].hash == hash &&
                _buckets[current_index].key == key) {
                return _buckets[current_index].value;
            }
            current_index = (current_index + 1) % _capacity;
            if (current_index == index)
				break; // Avoid infinite loop if table is full
        }

        return  std::nullopt;
    }

    bool remove(const key_t& key)
	{
        const uint64_t hash = hasher(key);
        size_t index = hash % _capacity;
        uint8_t group_hash = static_cast<uint8_t>(hash & HASH_MASK);

        // Check the initial group
        for (size_t offset = 0; offset < GROUP_SIZE; ++offset) {
            size_t probe_index = (index + offset) % _capacity;
            if ((_metadata[probe_index] & FULL) &&
                (_metadata[probe_index] & HASH_MASK) == group_hash &&
                _buckets[probe_index].hash == hash &&
                _buckets[probe_index].key == key) {
                // No need to explicitly clear the key string, destructor handles it
                _buckets[probe_index].value = value_t();
                _metadata[probe_index] = DELETED | group_hash;
                --_size;
                return true;
            }
        }

        // Linear probe if not found in the group
        size_t current_index = index;
        while (_metadata[current_index] & FULL) {
            if ((_metadata[current_index] & HASH_MASK) == group_hash &&
                _buckets[current_index].hash == hash &&
                _buckets[current_index].key == key) {
                _buckets[current_index].value = value_t();
                _metadata[current_index] = DELETED | group_hash;
                --_size;
                return true;
            }
            current_index = (current_index + 1) % _capacity;
            if (current_index == index) break;
        }

        return false;
    }
};

