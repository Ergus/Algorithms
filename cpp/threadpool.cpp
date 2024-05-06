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

#include "threadpool.hpp"

using namespace std::chrono_literals;

void myfunc(size_t i)
{
	std::thread::id tid = std::this_thread::get_id();
	size_t wid = my::threadpool_t::getWorkerId();

	std::cout << "Function: " << i << ":" << tid << ":" << wid << " start" << std::endl;
	std::this_thread::sleep_for(50ms);
	std::cout << "Function: " << i << ":" << tid << ":" << wid << " end" << std::endl;;
}

int main(int argc, char *argv[])
{
	argparser::init(argc, argv);
	const size_t ntasks = argparser::cl<size_t>("N Tasks");

	my::threadpool_t pool(4);

	for (size_t i = 0; i < ntasks; ++i)
		pool.pushTask(myfunc, i);

	for (size_t i = 0; i < ntasks; ++i)
		pool.pushTask(
			[](size_t i){
					size_t wid = my::threadpool_t::getWorkerId();

					std::cout << "Lambda: " << i << ":" << wid  << " start" << std::endl;
					std::this_thread::sleep_for(50ms);
					std::cout << "Lambda: " << i << ":" << wid << " end" << std::endl;;
			}, i);

	std::cout << "----- Start Waiting 1 -----" << std::endl;
	pool.taskWait();
	std::cout << "----- End Waiting 1 -----" << std::endl;

	std::vector<int> tmp1(100);
	std::iota(tmp1.begin(), tmp1.end(), 0);

	std::vector<int> tmp2(100);

	my::transform(pool, tmp1.begin(), tmp1.end(), tmp2.begin(),
	              [](int in) -> int {
					  size_t wid = my::threadpool_t::getWorkerId();
					  std::cout << "Trans: " << in << " " << wid << std::endl;
					  return in * in;
				  }
	);

	std::vector<int> pattern(100);
	std::transform(tmp1.begin(), tmp1.end(), pattern.begin(),
	               [](int in) -> int {
					   return in * in;
				   }
	);

	std::cout << "----- Start Waiting 2 -----" << std::endl;
	pool.taskWait();
	std::cout << "----- End Waiting 2 -----" << std::endl;
	myassert(pattern == tmp2);

	pool.setPolicy(1);
	my::transform(pool, tmp1.begin(), tmp1.end(), tmp2.begin(),
	              [](int in) -> int {
					  size_t wid = my::threadpool_t::getWorkerId();
					  std::cout << "Trans: " << in << " " << wid << std::endl;
					  return in * in;
				  }
	);

	std::cout << "----- Start Waiting 3 -----" << std::endl;
	pool.taskWait();
	std::cout << "----- End Waiting 3 -----" << std::endl;
	myassert(pattern == tmp2);

    return 0;
}

