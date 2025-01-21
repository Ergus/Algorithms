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

#include <iostream>
#include "apache_string.cuh"

const std::string text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod"
                         "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim "
                         "veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
                         "commodo consequat. Duis aute irure dolor in reprehenderit in voluptate "
                         "velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint "
                         "occaecat cupidatat non proident, sunt in culpa qui officia deserunt "
                         "mollit anim id est laborum.";

int main()
{
	std::cout << "Original string:" << std::endl;
	std::cout << text << std::endl;

	apache_string tmp_host(text);
	std::cout << "Apache string cpu:" << std::endl;
	std::cout << tmp_host << std::endl;

	apache_string tmp_device(text, 32);
	std::cout << "Apache string gpu:" << std::endl;
	std::cout << tmp_device << std::endl;

	myassert(tmp_host == tmp_device);
	return 0;
}
