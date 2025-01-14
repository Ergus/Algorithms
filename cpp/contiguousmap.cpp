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

#include <argparser.hpp>

#include "contiguousmap.hpp"
#include "utils.h"

int main()
{
	cmap<int, int> mymap(16);

	std::vector<std::pair<int, int>> reference = {{8, 16},
												  {4, 8},
												  {10, 20},
												  {2, 4},
												  {3, 6},
												  {7, 14},
												  {5, 10}};

	std::cout << "Init some values from reference" << std::endl;
	for (auto &[k, v] : reference) {
		std::cout << "init[" << k << "] = " << v << std::endl;
		mymap[k] = v;
	}

	std::cout << "\nPrint internal vector" << std::endl;
	for (const auto &node: mymap.get_internal_vector()){
		std::cout << node << std::endl;
	}

	std::cout << "\nCompare some values from reference" << std::endl;
	for (auto &[k, v] : reference) {
		myassert(mymap[k] == v);
	}

	std::cout << "\nIterate recursive" << std::endl;
	mymap.traverse_dfs(
		0,
		[](size_t i, const std::pair<int, int>& p)
		{
			std::cout << i << " -> " << p << std::endl;
		}
	);

	std::cout << "\nIterate non-recursive" << std::endl;
	mymap.traverse_dfs_it(
		0,
		[](size_t i, const std::pair<int, int>& p)
		{
			std::cout << i << " -> " << p << std::endl;
		}
	);

	std::cout << "\nIterate with iterator" << std::endl;
	for (const auto& it : mymap) {
		std::cout << it << std::endl;
	}

	return 0;
}
