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


	// Function to heapify a subtree rooted at node i using Floyd's method
	template <typename T>
	void heapifyFloyd(std::vector<T>& arr, size_t n, size_t i)
	{
		size_t current = i;
		size_t child = 2 * current + 1; // Left child

		while (child < n) {
			size_t right = child + 1;
			// Select the larger of the left and right children
			if (child + 1 < n && arr[child + 1] > arr[child]) {
				child = right;
			}

			// If the current node is already larger
			if (arr[current] >= arr[child]) {
				break;
			}

			// Swap the current node with the larger child
			std::swap(arr[current], arr[child]);

			// Move down the heap
			current = child;
			child = 2 * current + 1;
		}
	}

	// Function to perform Floyd's Heap Sort
	template<typename T>
	void heapSortFloyd(std::vector<T>& arr)
	{
		size_t n = arr.size();

		// Build the max-heap using Floyd's bottom-up method
		for (size_t i = n / 2; i > 0; --i) {
			heapifyFloyd(arr, n, i - 1);
		}

		// Extract elements from the heap one by one
		for (int i = n - 1; i > 0; --i) {
			std::swap(arr[0], arr[i]); // Move the root (largest element) to the end
			heapifyFloyd(arr, i, 0); // Restore the heap property for the reduced heap
		}
	}

}
