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


#include <compare>

template <typename T>
class iterator_t {
	T &_owner; /** Owner matrix */
	size_t _idx;    /** Iterator index */

public:
	iterator_t(T &owner, size_t idx = 0)
	: _owner(owner), _idx(idx)
	{
		assert(_idx <= _owner._rows);
	}

	iterator_t(const iterator_t &) = default;

	/** De-reference operator.

		Return the std::span<T> associated with the row pointed by the
		iterator_t.
	*/
	auto operator*()
	{
		return _owner[_idx];
	}

	iterator_t &operator+=(int step)
	{
		_idx += step;
		assert(_idx <= _owner._rows);
		return *this;
	}

	iterator_t &operator++()
	{
		return (*this) += 1;
	}

	iterator_t operator++(int)
	{
		iterator_t ret(*this);
		return ret += 1;
	}

	iterator_t operator+(int step) const
	{
		iterator_t ret(*this);
		return ret += step;
	}

	/** Decrement operators
		@param[in] step Distance to moce the iterator_t
	*/
	iterator_t &operator-=(int step)
	{
		return (*this) += (-step);
	}

	iterator_t &operator--()
	{
		return (*this) -= 1;
	}

	iterator_t operator--(int)
	{
		iterator_t ret(*this);
		return ret -= 1;
	}

	iterator_t operator-(int step) const
	{
		return (*this) + (-step);
	}

	auto operator<=>(const iterator_t &other) const
	{
		return _idx <=> other._idx;
	}

	bool operator==(const iterator_t& other) const
	{
		return _idx == other._idx;
	}
};
