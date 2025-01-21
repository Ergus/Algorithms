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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.	 If not, see <http://www.gnu.org/licenses/>.
 */

#pragma nv_diag_suppress 815 // suppress consr return warning

#include <iostream>
#include <random>
#include <argparser.hpp>

#include "apache_string.cuh"

const std::string default_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod"
						 "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim "
						 "veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
						 "commodo consequat. Duis aute irure dolor in reprehenderit in voluptate "
						 "velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint "
						 "occaecat cupidatat non proident, sunt in culpa qui officia deserunt "
						 "mollit anim id est laborum.";

//const std::string default_text = "Lorem ipsum dolor sit" ;


std::string random_string(std::size_t length)
{
	// I insert more spaces to increase the probability of inserting
	// spaces and emulate real words
	const std::string CHARACTERS = " 012345 6789 ABCDEF GHIJK LMNO PQRS TUVW XYZ abc def ghi jkl mno pqr stu vw xyz";
	std::string random_string;
	random_string.reserve(length);

	std::random_device random_device;
	std::mt19937 generator(random_device());
	std::uniform_int_distribution<> distribution(0, CHARACTERS.size() - 1);

	// I assume that the string is trimmed
	random_string.push_back('x');
	for (int i = 1; i < length - 1; ++i) {
		random_string.push_back(CHARACTERS[distribution(generator)]);
	}
	random_string.push_back('x');

	return random_string;
}

int main(int argc, char **argv)
{
	argparser::init(argc, argv);
	const size_t size = argparser::cl<int>("phrase size", 0);
	const size_t printlimit = argparser::cl<int>("print_limit", 512); // Limit to print the vectors

	std::string text = (size > 0) ? random_string(size) : default_text;

	std::cout << "Original string:" << std::endl;
	if (size <= printlimit)
		std::cout << text << std::endl;

	apache_string tmp_host = apache_string::factory<>(text);
	if (size <= printlimit)
		std::cout << "Apache string cpu:" << tmp_host << std::endl;

	apache_string tmp_device = apache_string::factory<32>(text);
	if (size <= printlimit)
		std::cout << "Apache string gpu:" << tmp_device << std::endl;

	myassert(tmp_host == tmp_device);
	return 0;
}
