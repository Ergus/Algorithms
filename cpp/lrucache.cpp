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

#include "utils.h"

#include <argparser.hpp>
#include <iostream>

#include "lrucache.hpp"

int main(int argc, char **argv)
{
	argparser::init(argc, argv);
	const size_t size = argparser::cl<int>("array_size");
	const size_t printlimit = argparser::cl<int>("print_limit",512); // Limit to print the vectors

	lrucache<int, std::string, 10> cache;
	std::cout << "Initial: " << cache << std::endl;

	for (int i = 0; i < 15; ++i) {
		cache.push(i, std::to_string(i));
		myassert(cache.size() <= 10);
	}
	std::cout << "Insert 0->14: " << cache << std::endl;
	// After inserting 15 elements we must only have 10
	myassert(cache.size() == 10);

	// Assert it removed the older elements when passed 10
	for (int i = 0; i < 5; ++i) {
		myassert(cache.find(i) == cache.end());
	}

	// Assert that the elements from 5->14 are there
	for (int i = 5; i < 15; ++i) {
		myassert(cache.find(i) != cache.end());
	}

	// test the operator[] (and register an access to 5)
	string val = cache[5];
	std::cout << "Get 5: " << cache << std::endl;
	myassert(val == "5");

	// push 1 element to assert 5 is preserved and 6 is removed.
	cache.push(20, "value");
	std::cout << "Push 20: " << cache << std::endl;
	myassert(cache.find(5) != cache.end());
	std::cout << "Find 5: " << cache << std::endl;
	myassert(cache.find(6) == cache.end());
	myassert(cache.size() == 10);

	// Remove the element 7
	cache.erase(7);
	std::cout << "Erase 7: " << cache << std::endl;
	myassert(cache.size() == 9);

	// Remove the element using iterator
	auto it = cache.find(11);
	std::cout << "Find 11: " << cache << std::endl;
	cache.erase(it);
	std::cout << "Erase it(11): " << cache << std::endl;
	myassert(cache.size() == 8);

	// Tests iteration
	for (auto it : cache)
		std::cout << it.first << "\t" << (std::string)it.second << std::endl;

	return 0;
}
