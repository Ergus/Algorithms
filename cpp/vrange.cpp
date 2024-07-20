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

#include <thread>
#include <utils.h>
#include <argparser.hpp>
#include <iostream>
#include <numeric>

#include "vrange.hpp"


int main(int argc, char *argv[])
{
	argparser::init(argc, argv);
	const size_t start = argparser::cl<size_t>("Start");
	const size_t count = argparser::cl<size_t>("Count");
	const int step = argparser::cl<int>("Step");

	vrange vrange1(start, count, step);

	size_t i = 0;
    for(auto var: vrange1) {
        std::cout << var << " == " << vrange1[i] << std::endl;
		myassert(var == vrange1[i]);
		++i;
	}

    return 0;
}

