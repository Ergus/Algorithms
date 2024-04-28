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

#include "threadpool.hpp"

using namespace std::chrono_literals;

void myfunc(size_t i)
{
	std::thread::id tid = std::this_thread::get_id();
	size_t wid = threadpool_t::get_worker().get_id();

	std::cout << "Function: " << tid << " : " << wid << " start" << std::endl;
	std::this_thread::sleep_for(500ms);
	std::cout << "Function: " << tid << " : " << wid << " end" << std::endl;;
}

int main(int argc, char *argv[])
{
	argparser::init(argc, argv);
	const size_t ntasks = argparser::cl<size_t>("N Tasks");

	threadpool_t pool(4);

	for (size_t i = 0; i < ntasks; ++i)
		pool.pushTask(myfunc, i);

	for (size_t i = 0; i < ntasks; ++i)
		pool.pushTask(
			[](size_t i){
					size_t wid = threadpool_t::get_worker().get_id();

					std::cout << "Lambda: " << wid  << " start" << std::endl;
					std::this_thread::sleep_for(500ms);
					std::cout << "Lambda: " << wid << " end" << std::endl;;
			}, i);

	std::cout << "----- Start Waiting -----" << std::endl;
	pool.taskWait();
	std::cout << "----- End Waiting -----" << std::endl;

    return 0;
}

