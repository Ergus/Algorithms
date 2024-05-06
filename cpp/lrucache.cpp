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
	const size_t limit = argparser::cl<int>("cache_size");

	lrucache<int, std::string> cache(limit);
	std::cout << "Initial: " << cache << std::endl;

	for (size_t i = 0; i < limit; ++i) {
		cache.push(i, std::to_string(i));
		myassert(cache.size() <= limit);
	}
	std::cout << "Insert 0->limit: " << cache << std::endl;
	myassert(cache.size() == limit);

	for (size_t i = limit; i < limit + 5; ++i) {
		cache.push(i, std::to_string(i));
		myassert(cache.size() == limit);
	}
	std::cout << "Insert limit-> limit+5: " << cache << std::endl;
	// After inserting 5 elements more we must only have limit elements
	myassert(cache.size() == limit);

	// Assert it overrided the first 5 elements
	for (size_t i = 0; i < 5; ++i) {
		myassert(cache.find(i) == cache.end());
	}

	// Assert that the elements from 5->limit are there
	// check all the elements to preserve cache order.
	for (size_t i = 5; i < limit + 5; ++i) {
		myassert(cache.find(i) != cache.end());
	}

	// test the operator[] (and register an access to 5)
	string val = cache[5];
	std::cout << "Get 5: " << cache << std::endl;
	myassert(val == "5");

	// push 1 more element to assert 5 is preserved and 6 is removed.
	cache.push(limit + 10, "limit+10");
	std::cout << "Push limit+10: " << cache << std::endl;
	myassert(cache.find(5) != cache.end());
	std::cout << "Find 5: " << cache << std::endl;
	myassert(cache.find(6) == cache.end());
	myassert(cache.size() == limit);

	// Remove the element 7
	cache.erase(7);
	std::cout << "Erase 7: " << cache << std::endl;
	myassert(cache.size() == limit-1);

	// Remove the element using iterator
	auto it = cache.find(11);
	std::cout << "Find 11: " << cache << std::endl;
	cache.erase(it);
	std::cout << "Erase it(11): " << cache << std::endl;
	myassert(cache.size() == limit-2);

	// Tests iteration
	for (auto it : cache)
		std::cout << it.first << "\t" << (std::string)it.second << std::endl;

	return 0;
}
