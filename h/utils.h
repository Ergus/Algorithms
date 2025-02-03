#pragma once
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

#include <span>
#include <fstream>
#include <vector>
#include <cassert>
#include <array>
#include <iostream>
#include <map>
#include <limits>
#include <cmath>

#define myassert(cond) {										\
		if (!(cond)) {											\
			fprintf(stderr, "%s%s:%u Assertion `%s' failed.\n", \
			        __func__, __FILE__, __LINE__, #cond);		\
			exit(EXIT_FAILURE);									\
		}														\
	}

/** Overload the << operator for std::pairs */
template <typename K, typename V>
std::ostream& operator<<(std::ostream &out, const std::pair<K,V> &pair)
{
	out << "{" << pair.first <<","<< pair.second << "} ";
	return out;
}

/** Overload the << operator for std::vectors and std::span */
template <typename T>
requires std::is_same_v<T, std::vector<typename T::value_type>>
		  || std::is_same_v<T, std::span<typename T::element_type>>
std::ostream& operator<<(std::ostream &out, const T &arr)
{
	for (const auto &it : arr)
		out << it << " ";

	out << std::endl;

	return out;
}

/** Overload the << operator for std::map */
template <typename K, typename V>
std::ostream& operator<<(std::ostream &out, const std::map<K,V> &map)
{
	out << "{ ";
	for (auto it : map)
		out << it;

	out << "}" << std::endl;

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

/**
   Counts the number of bits == 1 in the input.

   This function was added in C++-20, so, this function is actually
   not needed anymore.
 */
size_t popcount(unsigned int i)
{
	size_t count = 0;
	while (i) {
		i &= (i - 1);
		++count;
	}
	return count;
}

/**
   Get the closest power of two not bigger than input

   For example: given 9, 10 or 15, will will return 8;
   To get the next power of two bigger than input just multiply the
   result by two.
 */
unsigned int get_pow_2(unsigned int i)
{
	unsigned int tmp = i & (i - 1);
	while (tmp) {
		i = tmp;
		tmp = i & (i - 1);
	}
	return i;
}


std::vector<int> prime_factorization(int n)
{
    std::vector<int> result;

    // Handle factor 2 separately
    while (n % 2 == 0) {
        result.push_back(2);
        n /= 2;
    }

    // Check for odd factors
    for (int i = 3; i * i <= n; i += 2) {
        while (n % i == 0) {
            result.push_back(i);
            n /= i;
        }
    }

    // If n is still greater than 2, it's a prime number
    if (n > 2) {
        result.push_back(n);
    }

	return result;
}

std::pair<int, int> closest_divisors(int n)
{
	assert(n > 1);
    int sqrt_n = std::sqrt(n);
    for (int i = sqrt_n; i >= 1; --i) {
        if (n % i == 0) {
            return {i, n / i};
        }
    }
	assert(n == 1);
    return {1, n}; // Should never reach here for n >= 1
}
