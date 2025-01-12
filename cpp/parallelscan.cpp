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

#include <utils.h>

#include <argparser.hpp>
#include <iostream>
#include <random>

#include "parallelscan.hpp"

int main(int argc, char **argv)
{
	argparser::init(argc, argv);
	const size_t size = argparser::cl<int>("array_size");

	std::vector<int> v(size);
	std::iota(v.begin(), v.end(), 1);
	std::shuffle(v.begin(), v.end(), std::mt19937{std::random_device{}()});

	std::cout << v << std::endl;

	std::vector<int> v0(size);
	std::exclusive_scan(v.begin(), v.end(), v0.begin(), 0);
	std::cout << v0 << std::endl;

	std::vector<int> v1(size);
	my::exclusiveScanParallelForkJoin(v.begin(), v.end(), v1.begin(), 8);
	std::cout << v1 << std::endl;
	myassert(v0 == v1);

	std::vector<int> v2(size);
	my::exclusiveScanParallelSync(v.begin(), v.end(), v2.begin(), 8);
	std::cout << v2 << std::endl;
	myassert(v0 == v2);

	return 0;
}
