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

#include "matrix.hpp"

int main()
{
	std::cout << "Construct a zero filled matrix" << std::endl;
	matrix<double> mat(5, 4);

	std::cout << "\nInitialize usng the [] operator" << std::endl;
	for (size_t r = 0, count = 0; r < 5; ++r)
		for (size_t c = 0; c < 4; ++c)
			mat[r][c] = (double) count++;

	std::cout << "\nPrint the matrix using the overloaded operator <<" << std::endl;
	std::cout << mat;

	// Test the accessors
	for (size_t r = 0, count = 0; r < 5; ++r) {
		for (size_t c = 0; c < 4; ++c) {
			double tmp = (double) count++;
			myassert(mat[r][c] == tmp);
			myassert(mat(r, c) == tmp);
		}
	}

	// Print C++-11 for syntax dual iterator
	// Test dual iterator accesor.
	std::cout << "\nTest two iteration levels access and print" << std::endl;
	size_t i = 0;
	for (const auto it : mat) {
		size_t j = 0;
		for (const auto itt : it) {
			std::cout << itt << " ";
			myassert(mat[i][j++] == itt);
		}
		std::cout << std::endl;
		++i;
	}

	std::cout << "\nTest one iteration levels access print" << std::endl;
	for (const auto it : mat)
		std::cout << it;

	std::cout << "\nChange the values with for loop and implicit iterator" << std::endl;
	for (auto it : mat)
		for (auto &val : it)
			val = 42;

	std::cout << "\nTest values after for loop and implicit iterator" << std::endl;
	for (size_t r = 0; r < 5; ++r) {
		for (size_t c = 0; c < 4; ++c) {
			myassert(mat[r][c] == 42);
			myassert(mat(r, c) == 42);
		}
	}

	std::cout << "\nTest modification with iterator" << std::endl;
	for (matrix<double>::iterator it =  mat.begin(); it != mat.end(); ++it)
		for (auto itt = (*it).begin(); itt != (*it).end(); ++itt)
			*itt = -42;

	std::cout << "\nTest values after modification with iterator" << std::endl;
	for (size_t r = 0; r < 5; ++r) {
		for (size_t c = 0; c < 4; ++c) {
			myassert(mat[r][c] == -42);
			myassert(mat(r, c) == -42);
		}
	}

	std::cout << "\nPrint using const iterator" << std::endl;
	for (matrix<double>::const_iterator it =  mat.begin(); it != mat.end(); ++it)
		std::cout << *it;

	return 0;
}
