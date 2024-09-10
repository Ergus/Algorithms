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
	T &_owner;      /** Owner matrix */
	size_t _idx;    /** Iterator index */

public:
	/** Constructor
		@param[in] owner Matrix owner for this iterator it is a reference
		because an iterator never changes the owner
		@param[in] idx Starting row for the iterator
	*/
	iterator_t(T &owner, size_t idx = 0)
	: _owner(owner), _idx(idx)
	{
		assert(_idx <= _owner._rows);
	}

	/** Use default copy constructor */
	iterator_t(const iterator_t &) = default;

	/** De-reference operator.

		Return the std::span<T> associated with the row pointed by the
		iterator_t.
		This implementation used auto as a return type because the
		const_iterator version returns span<const T> while the non
		const implements two alternatives.
		Implementing these versions manually implies duplicate the code
		and use `require` + is_const. This alternative is the same in
		a much more compact code.
	*/
	auto operator*()
	{
		return _owner[_idx];
	}

	/** \defgroup Increment operators
	 *  @{
	 */

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
	/**@}*/


	/** \defgroup Decrement operators
	 *  @{
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
	/**@}*/

	/** \defgroup Comparison operators
	 *  @{
	 */
	auto operator<=>(const iterator_t &other) const
	{
		assert(&_owner == &other._owner);
		return _idx <=> other._idx;
	}

	bool operator==(const iterator_t& other) const
	{
		assert(&_owner == &other._owner);
		return _idx == other._idx;
	}
	/**@}*/
};
