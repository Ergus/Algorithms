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
#include <functional>
#include <array>

namespace my {

	template <typename Iter, typename Oter>
	void exclusiveScanParallel(
		Iter first, Iter last, Oter ofirst, size_t nThreads = 0
	) {
		using type = typename Iter::value_type;

		if (nThreads == 0)
			nThreads = std::thread::hardware_concurrency();

		const size_t size = last - first;

		if (size < 2)
			return;

		if (size < 32) {
			std::exclusive_scan(first, last, ofirst, 0);
			return;
		}

		const std::vector<std::array<size_t,2>> ranges = computeRanges(size, nThreads);

		std::vector<std::thread> threads(nThreads);

		// First scan by chunks
		for (size_t i = 0; i < nThreads; ++i) {
			threads[i] = std::thread(
				std::exclusive_scan<Iter, Oter, type>,
				first + ranges[i][0],
				first + ranges[i][0] + ranges[i][1],
				ofirst + ranges[i][0],
				type()
			);
		}

		for (auto &thread : threads)
			thread.join();

		// Scan the last indices for arrays
		std::vector<size_t> count(nThreads);
		for (size_t i = 1; i < nThreads; ++i) {
			const size_t lastIdx = ranges[i - 1][0] + ranges[i - 1][1] - 1;
			count[i] = count[i - 1] + *(ofirst + lastIdx) + *(first + lastIdx);
		}

		for (size_t i = 0; i < nThreads; ++i) {
			threads[i] = std::thread(
				std::for_each<Oter, std::function<void(typename Oter::reference)>>,
				ofirst + ranges[i][0],
				ofirst + ranges[i][0] + ranges[i][1],
				[value = count[i]](typename Oter::reference v)
				{
					v += value;
				}
			);
		}

		for (auto &thread : threads)
			thread.join();
	}

}
