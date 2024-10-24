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

#pragma once

#include <algorithm>
#include <numeric>
#include <vector>
#include <thread>
#include <atomic>

#include <cassert>
#include <cstdlib>

#include <iostream>

namespace my {

// Quick sort in C++

	template <typename T>
	size_t partition(std::vector<T> &arr, size_t left, size_t right)
	{
		const T pivot = arr[right]; // right most
		size_t i = left - 1; // greater element pos

		// traverse each element compare them with the pivot
		for (size_t j = left; j < right; j++) {
			if (arr[j] > pivot)
				continue;

			std::swap(arr[++i], arr[j]);
		}

		std::swap(arr[i + 1], arr[right]); // swap pivot with the greater element at i

		return (i + 1); // return the partition point
	}

	template <typename T>
	void quickSort(std::vector<T> &arr, size_t left, size_t right)
	{
		if (left >= right)
			return;

		const size_t pivot = partition(arr, left, right - 1);

		quickSort(arr, left, pivot);

		quickSort(arr, pivot + 1, right);
	}

}
