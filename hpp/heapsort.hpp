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
#include <vector>
#include <cassert>
#include <cstdlib>

namespace my {

	template <typename T>
	void heapify(std::vector<T> &arr, size_t n, size_t i)
	{
		// i => root
		// n => size

		// Find largest among root, left child and right child
		size_t largest = i;
		size_t left = 2 * i + 1;
		size_t right = 2 * i + 2;

		if (left < n && arr[left] > arr[largest])
			largest = left;

		if (right < n && arr[right] > arr[largest])
			largest = right;

		// Swap and continue heapifying if root is not largest
		if (largest != i) {
			std::swap(arr[i], arr[largest]);
			heapify(arr, n, largest);
		}
	}

	template <typename T>
	void heapSort(std::vector<T> &arr)
	{
		// Build max heap
		for (int i = arr.size() / 2 - 1; i >= 0; --i)
			heapify(arr, arr.size(), (size_t) i);

		// Heap sort
		for (int i = arr.size() - 1; i >= 0; --i) {
			std::swap(arr[0], arr[i]);
			heapify(arr, (size_t) i, 0);
		}
	}
}
