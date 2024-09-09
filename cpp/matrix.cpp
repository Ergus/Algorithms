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


	// Construct a zero filled matrix
	matrix<double> mat(5, 4);

	// Initialize usng the [] operator
	for (size_t r = 0, count = 0; r < 5; ++r)
		for (size_t c = 0; c < 4; ++c)
			mat[r][c] = (double) count++;


	// Print the matrix using the overloaded operator<<
	std::cout << mat << std::endl;

	// Initialize usng the [] operator
	for (size_t r = 0, count = 0; r < 5; ++r) {
		for (size_t c = 0; c < 4; ++c) {
			double tmp = (double) count++;
			assert(mat[r][c] == tmp);
			assert(mat(r, c) == tmp);
		}
	}

	return 0;
}
