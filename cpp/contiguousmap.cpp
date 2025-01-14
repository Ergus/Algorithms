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

	// Init some values
	mymap[8] = 16;
	mymap[4] = 8;
	mymap[10] = 20;
	mymap[2] = 4;
	mymap[3] = 6;
	mymap[7] = 14;
	mymap[5] = 10;

	mymap.printvec();

	myassert(mymap[8] == 16);
	myassert(mymap[4] == 8);
	myassert(mymap[10] == 20);
	myassert(mymap[2] == 4);
	myassert(mymap[3] == 6);
	myassert(mymap[7] == 14);
	myassert(mymap[5] == 10);

	std::cout << "Iterate recursive" << std::endl;
	mymap.traverse_dfs(
		0,
		[](size_t i, const std::pair<int, int>& p)
		{
			std::cout << i << " -> " << p << std::endl;
		}
	);

	std::cout << std::endl;

	std::cout << "Iterate non-recursive" << std::endl;
	mymap.traverse_dfs_it(
		0,
		[](size_t i, const std::pair<int, int>& p)
		{
			std::cout << i << " -> " << p << std::endl;
		}
	);

	std::cout << "Iterate with iterator" << std::endl;
	for (const auto& it : mymap) {
		std::cout << it << std::endl;
	}

	return 0;
}
