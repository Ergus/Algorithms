// Copyright (C) 2024  Jimmy Aguilar Mena

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
#include <iostream>
#include "rangemap.hpp"

int main(int argc, char **argv)
{
	rangeMap<int, int> map(0);
	myassert(map.isEqual({{-2147483648,0}}));

	// Insertions no overlap
	// First insertion
	map.setRange(0, 100, 1);
	std::cout << map << std::endl;
	myassert(map.isEqual({{-2147483648,0},{0,1},{100,0}}));

	// Insert on the right of that one
	map.setRange(100, 200, 2);
	std::cout << map << std::endl;
	myassert(map.isEqual( { {-2147483648,0}, {0,1}, {100,2}, {200,0} }));

	// Insert on the left of everything
	map.setRange(-100, 0, 3);
	std::cout << map << std::endl;
	myassert(map.isEqual({{-2147483648,0}, {-100,3}, {0,1}, {100,2}, {200,0}}));

	// Overlap insertion
	// Insert within a range
	map.setRange(40, 50, 4);
	std::cout << map << std::endl;
	myassert(map.isEqual({ {-2147483648,0}, {-100,3}, {0,1}, {40,4}, {50,1}, {100,2}, {200,0} }));

	// Insert within a range in start
	map.setRange(50, 60, 5);
	std::cout << map << std::endl;
	myassert(map.isEqual({ {-2147483648,0}, {-100,3}, {0,1}, {40,4}, {50,5}, {60,1}, {100,2}, {200,0} }));

	// Full overlap 1 range
	map.setRange(40, 50, 6);
	std::cout << map << std::endl;
	myassert(map.isEqual({ {-2147483648,0}, {-100,3}, {0,1}, {40,6}, {50,5}, {60,1}, {100,2}, {200,0} }));

	// Full overlap 2 ranges
	map.setRange(40, 60, 7);
	std::cout << map << std::endl;
	myassert(map.isEqual({ {-2147483648,0}, {-100,3}, {0,1}, {40,7}, {60,1}, {100,2}, {200,0} }));

	// Partial overlap 2 ranges
	map.setRange(50, 70, 8);
	std::cout << map << std::endl;
	myassert(map.isEqual({ {-2147483648,0}, {-100,3}, {0,1}, {40,7}, {50,8}, {70,1}, {100,2}, {200,0} }));

	// Partial overlap 3 ranges
	map.setRange(45, 75, 9);
	std::cout << map << std::endl;
	myassert(map.isEqual({ {-2147483648,0}, {-100,3}, {0,1}, {40,7}, {45,9}, {75,1}, {100,2}, {200,0} }));

	// Partial overlap 2 ranges one at the end
	map.setRange(150, 250, 10);
	std::cout << map << std::endl;
	myassert(map.isEqual({ {-2147483648,0}, {-100,3}, {0,1}, {40,7}, {45,9}, {75,1}, {100,2}, {150,10}, {250,0} }));

	// Insert an interval just next to other but with same value (merge left)
	map.setRange(250, 300, 10);
	std::cout << map << std::endl;
	myassert(map.isEqual({ {-2147483648,0}, {-100,3}, {0,1}, {40,7}, {45,9}, {75,1}, {100,2}, {150,10}, {300,0} }));

	// Insert an interval just before another with same value (merge right)
	map.setRange(20, 40, 7);
	std::cout << map << std::endl;
	myassert(map.isEqual({ {-2147483648,0}, {-100,3}, {0,1}, {20,7}, {45,9}, {75,1}, {100,2}, {150,10}, {300,0} }));

	// Insert an interval with same value to merge latter
	map.setRange(75, 110, 10);
	std::cout << map << std::endl;
	myassert(map.isEqual({ {-2147483648,0}, {-100,3}, {0,1}, {20,7}, {45,9}, {75,10}, {110,2}, {150,10}, {300,0} }));

	// Insert an interval with merge in both sides
	map.setRange(110, 150, 10);
	std::cout << map << std::endl;
	myassert(map.isEqual({ {-2147483648,0}, {-100,3}, {0,1}, {20,7}, {45,9}, {75,10}, {300,0} }));

	// Insert embeded interval with same value then parent
	map.setRange(-75, -50, 3);
	std::cout << map << std::endl;
	myassert(map.isEqual({ {-2147483648,0}, {-100,3}, {0,1}, {20,7}, {45,9}, {75,10}, {300,0} }));

	// Insert left interval to merge
	map.setRange(-200, -150, 3);
	std::cout << map << std::endl;
	myassert(map.isEqual({ {-2147483648,0}, {-200,3}, {-150,0}, {-100,3}, {0,1}, {20,7}, {45,9}, {75,10}, {300,0} }));

	// Insert left interval to merge
	map.setRange(-300, -250, 3);
	std::cout << map << std::endl;
	myassert(map.isEqual({ {-2147483648,0}, {-300,3}, {-250,0}, {-200,3}, {-150,0}, {-100,3}, {0,1}, {20,7}, {45,9}, {75,10}, {300,0} }));

	// Insert triple merge interval
	map.setRange(-275, -50, 3);
	std::cout << map << std::endl;
	myassert(map.isEqual({ {-2147483648,0}, {-300,3}, {0,1}, {20,7}, {45,9}, {75,10}, {300,0} }));

	// Partial overlay
	map.setRange(-300, -200, 4);
	std::cout << map << std::endl;
	myassert(map.isEqual({ {-2147483648,0}, {-300,4}, {-200,3}, {0,1}, {20,7}, {45,9}, {75,10}, {300,0} }));

	return 0;
}
