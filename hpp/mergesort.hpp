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

	// Indices
	template <typename T>
	void merge(std::vector<T> &arr, size_t l, size_t c, size_t r)
	{
		std::vector<int> L(arr.begin() + l, arr.begin() + c);
		std::vector<int> M(arr.begin() + c, arr.begin() + r);

		// Maintain current index of sub-arrays and main array
		size_t i = 0, j = 0, k = l;

		// Until we reach either end of either L or M, pick larger among
		// elements L and M and place them in the correct position at A[p..r]
		while (i < L.size() && j < M.size())
			arr[k++] = (L[i] <= M[j]) ? L[i++] : M[j++];

		// When we run out of elements in either L or M,
		// pick up the remaining elements and put in A[p..r]
		while (i < L.size())
			arr[k++] = L[i++];

		while (j < M.size())
			arr[k++] = M[j++];
	}


	template <typename T>
	void mergeSort(std::vector<T> &arr, size_t l, size_t r)
	{
		assert(r - l > 1);
		assert(r <= arr.size());

		// m is the point where the array is divided into two subarrays
		const size_t m = (l + r) / 2;

		if (m - l > 1)
			mergeSort(arr, l, m);

		if (r - m > 1)
			mergeSort(arr, m, r);

		merge(arr, l, m, r);
	}

	// Iterators
	template <typename T>
	void mergeIterator(T start, T m, T end)
	{
		if (start == m || m == end)
			return;

		std::vector<typename T::value_type> L(start, m);
		std::vector<typename T::value_type> R(m, end);

		// Maintain current index of sub-arrays and main array
		T itL = L.begin();
		T itR = R.begin();

		// Until we reach either end of either L or M, pick larger among
		// elements L and M and place them in the correct position at A[p..r]
		while (itL != L.end() && itR != R.end())
			*start++ = (*itL <= *itR) ? *itL++ : *itR++;

		// When we run out of elements in either L or M,
		// pick up the remaining elements and put in A[p..r]
		while (itL != L.end())
			*start++ = *itL++;

		while (itR != R.end())
			*start++ = *itR++;
	}

	template <typename T>
	void mergeSortIterator(T start, T end)
	{
		assert(start != end);

		const size_t size = std::distance(start, end);
		assert (size > 1);

		// m is the point where the array is divided into two subarrays
		T m = start + size / 2;

		if (std::distance(start, m) > 1)
			mergeSortIterator(start, m);

		if (std::distance(m, end) > 1)
			mergeSortIterator(m, end);

		mergeIterator(start, m, end);
	}

	// std
	template<class Iter>
	void mergeSortStd(Iter first, Iter last)
	{
		if (last - first < 2)
			return;

		Iter middle = first + (last - first) / 2;
		mergeSortStd(first, middle);
		mergeSortStd(middle, last);

		std::inplace_merge(first, middle, last);
	}

	// parallel
	template<class Iter>
	void mergeSortStdParallel(Iter first, Iter last, size_t nThreads = 0)
	{
		if (nThreads == 0)
			nThreads = std::thread::hardware_concurrency();

		static std::atomic<size_t> counter = 0; // count the number of created threads

		const size_t size = last - first;

		if (size < 2)
			return;

		if (size < 32) {
			mergeSortStd(first, last);
			return;
		}

		Iter middle = first + (last - first) / 2;

		std::thread t;

		if (counter++ < nThreads) {
			t = std::thread(mergeSortStdParallel<Iter>, first, middle, nThreads); 
		} else {
			mergeSortStd(first, middle, nThreads);
		}

		if (std::distance(middle, last) > 1)
			mergeSortStdParallel(middle, last, nThreads);

		if (t.joinable()) {
			t.join();
			--counter;
		}

		std::inplace_merge(first, middle, last);

		std::cout << counter << std::endl;
	}
}
