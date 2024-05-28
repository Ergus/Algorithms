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

#include <vector>
#include <string>
#include <fstream>

class apache_string {
	std::string buffer;
	std::vector<size_t> sizes;

public:
	apache_string(const std::string &input)
	{
		sizes.push_back(0);
		for(char c : input) {
			if (c == ' ')
				sizes.push_back(buffer.size());
			else
				buffer.push_back(c);
		}
		sizes.push_back(buffer.size());
	}

	friend std::ostream &operator <<(std::ostream &out, const apache_string &str)
	{
		for (auto it = str.sizes.begin(); std::next(it) != str.sizes.end(); std::advance(it, 1))
		{
			out << str.buffer.substr(*it, *std::next(it) - *it) << " ";
		}
		return out;
	}

};
