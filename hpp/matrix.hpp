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

#include "utils.h"

template <typename T>
class matrix {

public:

	matrix(size_t rows, size_t cols)
	: _rows(rows), _cols(cols), _mat(rows * cols)
	{
	}


	std::span<T> operator[](size_t idx)
	{
		if (idx < _rows)
			return std::span<T>(_mat.begin() + idx * _cols, _cols);

		return std::span<T>();
	}


	std::span<const T> operator[](size_t idx) const
	{
		if (idx < _rows)
			return std::span<const T>(_mat.begin() + idx * _cols, _cols);

		return std::span<const T>();
	}


	friend std::ostream& operator<<(std::ostream& out, const matrix& m)
	{
		for (size_t r = 0; r < m._rows; ++r)
		 	out << m[r];

		return out;
	}


private:

	size_t _rows, _cols;
	std::vector<T> _mat;

};
