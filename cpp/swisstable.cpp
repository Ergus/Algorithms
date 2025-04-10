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

// --- Example Usage ---
int main()
{
    SwissTable<std::string, int> ht;

    ht.insert("apple", 1);
    ht.insert("banana", 2);
    ht.insert("cherry", 3);
    ht.insert("date", 4);
    ht.insert("elderberry", 5);
    ht.insert("fig", 6);
    ht.insert("grape", 7);
    ht.insert("kiwi", 8);
    ht.insert("lemon", 9);
    ht.insert("mango", 10);
    ht.insert("nectarine", 11);
    ht.insert("orange", 12);
    ht.insert("pear", 13);
    ht.insert("quince", 14);
    ht.insert("raspberry", 15);
    ht.insert("strawberry", 16);
    ht.insert("tangerine", 17);

    std::cout << "Size: " << ht.size() << ", Capacity: " << ht.capacity() << std::endl;

    std::cout << "Value of 'banana': " << ht.find("banana").value_or(-1) << std::endl;
    std::cout << "Value of 'grape': " << ht.find("grape").value_or(-1) << std::endl;
    std::cout << "Value of 'watermelon': " << ht.find("watermelon").value_or(-1) << std::endl;

    ht.remove("banana");
    std::cout << "Value of 'banana' after removal: " << ht.find("banana").value_or(-1) << std::endl;

    ht.insert("blueberry", 18);
    std::cout << "Value of 'blueberry': " << ht.find("blueberry").value_or(-1) << std::endl;

    return 0;
}
