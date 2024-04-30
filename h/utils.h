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

#include <fstream>
#include <vector>
#include <cassert>
#include <array>
#include <iostream>

#define myassert(cond) {										\
		if (!(cond)) {											\
			fprintf(stderr, "%s%s:%u Assertion `%s' failed.\n", \
			        __func__, __FILE__, __LINE__, #cond);		\
			exit(EXIT_FAILURE);									\
		}														\
	}

/** Overload the << operator for vectors */
template <typename T>
std::ostream& operator<<(std::ostream &out, const std::vector<T> &arr)
{
	for (auto it : arr)
		out << it << " ";

	out << std::endl;

	return out;
}

/** Compute range pairs <start,end> for a container.
	@param[in] start Range start iterator
	@param[in] end Range end iterator
	@param[in] nThreads Number of subgroups to distribute into.
	@return a vector with pairs <start,size> for every interval.
*/
template <typename Iter>
std::vector<std::array<Iter, 2>> computeRanges(Iter start, Iter end, size_t nThreads)
{
	const std::vector<std::array<size_t,2>> ranges
		= computeRanges(std::distance(start, end), nThreads);

	std::vector<std::array<Iter, 2>> result(nThreads);

	for (size_t i = 0; i < nThreads; ++i)
		result[i] = {start + ranges[i][0], start + ranges[i][0] + ranges[i][1]};

	assert(end == result.last()[1]);

	return result;
}

/** Compute balanced pairs <start,size> for number of elements.
   @param[in] size Total size to distribute
   @param[in] nThreads Number of subgroups to distribute into.
   @return a vector with pairs <start,size> for every interval.
*/
std::vector<size_t> computeRanges(size_t size, size_t nThreads)
{
	const size_t quot = size / nThreads;
	const size_t rem = size % nThreads;

	std::vector<size_t> result(nThreads + 1);
	size_t acc = 0;
	for (size_t i = 0; i < nThreads; ++i)
	{
		result[i] = acc;
		acc += (quot + (i < rem));
	}
	result[nThreads] = acc;

	return result;
}

std::vector<size_t> computeChunks(size_t size, size_t chunkSize)
{
	const size_t nChunks = size / chunkSize;

	const size_t rem =  size % chunkSize;

	std::vector<size_t> result(nChunks + (int)(rem) + 1);
	size_t acc = 0;
	for (size_t i = 0; i < nChunks; ++i)
	{
		result[i] = acc;
		acc += chunkSize;
	}

	result[nChunks] = acc;
	if (rem)
		result[nChunks + 1] = rem;

	return result;
}
