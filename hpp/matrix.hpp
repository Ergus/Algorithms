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

	class iterator {
		matrix &_owner;
		size_t _idx;

		std::span<T> operator*()
		{
			return _owner[_idx];
		}
	};

	/** Construct zero filled matrix.
		@params[in] rows Number of rows
		@params[in] cols Number of columns
	*/
	matrix(size_t rows = 0, size_t cols = 0)
	: _rows(rows), _cols(cols), _mat(rows * cols)
	{
	}

	/** Row access operator.
		@params[in] idx the row index to return
		@return an std::span of the row
	*/
	std::span<T> operator[](size_t idx)
	{
		if (idx < _rows)
			return std::span<T>(_mat.begin() + idx * _cols, _cols);

		return std::span<T>();
	}

	/** Row access const operator.
		@params[in] idx the row index to return
		@return a const std::span of the row
	*/
	std::span<const T> operator[](size_t idx) const
	{
		if (idx < _rows)
			return std::span<const T>(_mat.begin() + idx * _cols, _cols);

		return std::span<const T>();
	}

	/** Use alternative operator to access elements
		@params[in] row row element to access
		@params[in] col column element to access
		@return element value to return
	*/
	T operator()(size_t row, size_t col)
	{
		assert(row < _rows);
		assert(col < _cols);
		return _mat[row * _cols + col];
	}

	T &operator()(size_t row, size_t col) const
	{
		assert(row < _rows);
		assert(col < _cols);
		return _mat[row * _cols + col];
	}

	/** Stream operator
		Overload for the operator<< for the matrix class
	*/
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
