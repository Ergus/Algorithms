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

#include "mergesort.hpp"

int main(int argc, char **argv)
{
	argparser::init(argc, argv);
	const size_t size = argparser::cl<int>("array_size");
	const size_t printlimit = argparser::cl<int>("print_limit",512); // Limit to print the vectors

	std::vector<int> v(size);
	std::iota(v.begin(), v.end(), 1);
	std::shuffle(v.begin(), v.end(), std::mt19937{std::random_device{}()});

	if (size <= printlimit)
		std::cout << "Intial:\n" << v << std::endl;

	std::vector<int> stl = v;
	std::sort(stl.begin(), stl.end());
	if (size <= printlimit)
		std::cout << "stl:\n" << stl << std::endl;
	myassert(std::is_sorted(stl.begin(), stl.end())) 

	std::vector<int> basic = v;
	my::mergeSort(basic, 0, basic.size());
	if (size <= printlimit)
		std::cout << "Basic:\n" << basic << std::endl;
	myassert(std::is_sorted(basic.begin(), basic.end()));
	myassert(basic == stl);

	std::vector<int> iter = v;
	my::mergeSortIterator(iter.begin(), iter.end());
	if (size <= printlimit)
		std::cout << "Iter:\n" << iter << std::endl;
	myassert(std::is_sorted(iter.begin(), iter.end()));
	myassert(iter == stl);
	
	std::vector<int> stdv = v;
	my::mergeSortStd(stdv.begin(), stdv.end());
	if (size <= printlimit)
		std::cout << "Std:\n" << stdv << std::endl;
	myassert(std::is_sorted(stdv.begin(), stdv.end()));
	myassert(stdv == stl);

	std::vector<int> parallel4 = v;
	my::mergeSortStdParallel(parallel4.begin(), parallel4.end(), 4);
	if (size <= printlimit)
		std::cout << "parallel4:\n" << parallel4 << std::endl;
	myassert(std::is_sorted(parallel4.begin(), parallel4.end()));
	myassert(parallel4 == stl);

	std::vector<int> parallel8 = v;
	my::mergeSortStdParallel(parallel8.begin(), parallel8.end(), 4);
	if (size <= printlimit)
		std::cout << "parallel4:\n" << parallel8 << std::endl;
	myassert(std::is_sorted(parallel8.begin(), parallel8.end()));
	myassert(parallel8 == stl);

	return 0;
}
