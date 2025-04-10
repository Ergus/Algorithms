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

#include <iostream>
#include <cstdint>

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

#include "swisstable.hpp"
#include "utils.h"

// --- Example Usage ---
int main()
{
	const std::vector<std::pair<std::string, int>> datas = {
		{"apple", 1},
		{"banana", 2},
		{"cherry", 3},
		{"date", 4},
		{"elderberry", 5},
		{"fig", 6},
		{"grape", 7},
		{"kiwi", 8},
		{"lemon", 9},
		{"mango", 10},
		{"nectarine", 11},
		{"orange", 12},
		{"pear", 13},
		{"quince", 14},
		{"raspberry", 15},
		{"strawberry", 16},
		{"tangerine", 17}
	};


	bool inserted = false;

	SwissTable<std::string, int> ht;

	// Test insert
	for (const std::pair<std::string, int> &data : datas) {

		inserted = ht.set(data.first, data.second);
		myassert(inserted);
	}

	// Tests size and capacity
	std::cout << "Size: " << ht.size() << ", Capacity: " << ht.capacity() << std::endl;
	myassert(ht.size() == 17);
	myassert(ht.capacity() == 32);

	// Test find existing
	myassert(ht.find("banana").has_value());
	std::cout << "Value of 'banana': " << ht.find("banana").value() << std::endl;
	myassert(ht.find("banana").value() == 2);

	myassert(ht.find("grape").has_value());
	std::cout << "Value of 'grape': " << ht.find("grape").value() << std::endl;
	myassert(ht.find("grape").value() == 7);

	// Test find non existing
	std::cout << "Value of 'watermelon': " << ht.find("watermelon").value_or(-1) << std::endl;
	myassert(!ht.find("watermelon").has_value());

	// Test remove a value
	bool removed = ht.remove("banana");
	myassert(removed);
	std::cout << "Value of 'banana' after removal: " << ht.find("banana").value_or(-1) << std::endl;
	myassert(!ht.find("banana").has_value());

	// Test insert after access
	inserted = ht.set("blueberry", 18);
	myassert(inserted);
	myassert(ht.find("blueberry").has_value());
	std::cout << "Value of 'blueberry': " << ht.find("blueberry").value() << std::endl;
	myassert(ht.find("blueberry").value() == 18);

	// Test reinsert Banana
	inserted = ht.set("banana", 20);
	myassert(inserted);
	myassert(ht.find("banana").has_value());
	std::cout << "Value of 'banana': " << ht.find("banana").value() << std::endl;
	myassert(ht.find("banana").value() == 20);

	// Test updates
	// Update Blueberry
	inserted = ht.set("blueberry", 20);
	myassert(!inserted);
	std::cout << "Value of updated 'blueberry': " << ht.find("blueberry").value() << std::endl;
	myassert(ht.find("blueberry").value() == 20);

	// Test update again inplace
	myassert(ht.find("banana").has_value());
	inserted = ht.set("banana", 21);
	myassert(!inserted);
	myassert(ht.find("banana").has_value());
	std::cout << "Value of 'banana' after removal: " << ht.find("banana").value() << std::endl;
	myassert(ht.find("banana").has_value());
	myassert(ht.find("banana").value() ==  21);


	return 0;
}
