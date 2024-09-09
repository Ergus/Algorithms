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
#include "iterator_t.hpp"

template <typename T>
class matrix {

public:

	friend class iterator_t<matrix>;
	friend class iterator_t<const matrix>;

	using value_type = T;
	using iterator = iterator_t<matrix>;
	using const_iterator = iterator_t<const matrix>;

	/** Construct zero filled matrix.
		@param[in] rows Number of rows
		@param[in] cols Number of columns
	*/
	matrix(size_t rows = 0, size_t cols = 0)
	: _rows(rows), _cols(cols), _mat(rows * cols)
	{
	}

	/** Row access operator.
		@param[in] idx the row index to return
		@return an std::span of the row
	*/
	std::span<T> operator[](size_t idx)
	{
		if (idx < _rows)
			return std::span<T>(_mat.begin() + idx * _cols, _cols);

		return std::span<T>();
	}

	/** Row access const operator.
		@param[in] idx the row index to return
		@return a const std::span of the row
	*/
	std::span<const T> operator[](size_t idx) const
	{
		if (idx < _rows)
			return std::span<const T>(_mat.begin() + idx * _cols, _cols);

		return std::span<const T>();
	}

	/** Use alternative operator to access elements
		@param[in] row row element to access
		@param[in] col column element to access
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

	const_iterator begin() const
	{
		return const_iterator(*this, 0);
	}

	const_iterator end() const
	{
		return const_iterator(*this, _rows);
	}

private:

	size_t _rows, _cols;
	std::vector<T> _mat;

};
