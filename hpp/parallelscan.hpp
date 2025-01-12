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
#include <functional>
#include <atomic>
#include <cassert>
#include <cstdlib>

#include "utils.h"

namespace my {

	template <typename Iter, typename Oter>
	void exclusiveScanParallelForkJoin(
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

		const std::vector<size_t> ranges = computeRanges(size, nThreads);

		std::vector<std::thread> threads(nThreads);

		// First scan by chunks
		for (size_t i = 0; i < nThreads; ++i) {
			threads[i] = std::thread(
				std::exclusive_scan<Iter, Oter, type>,
				first + ranges[i],
				first + ranges[i + 1],
				ofirst + ranges[i],
				type()
			);
		}

		for (auto &thread : threads)
			thread.join();

		// Scan the last indices for arrays
		std::vector<size_t> count(nThreads);
		for (size_t i = 1; i < nThreads; ++i) {
			const size_t lastIdx = ranges[i] - 1;
			count[i] = count[i - 1] + *(ofirst + lastIdx) + *(first + lastIdx);
		}

		// Update the chunks indices with the scanned values
		for (size_t i = 0; i < nThreads; ++i) {
			threads[i] = std::thread(
				std::for_each<Oter, std::function<void(typename Oter::reference)>>,
				ofirst + ranges[i],
				ofirst + ranges[i + 1],
				[value = count[i]](typename Oter::reference v)
				{
					v += value;
				}
			);
		}

		for (auto &thread : threads)
			thread.join();
	}

	template <typename Iter, typename Oter>
	void exclusiveScanParallelSync(
		Iter first, Iter last, Oter ofirst, size_t nThreads = 0
	) {
		using type = typename Iter::value_type;

		if (nThreads == 0)
			nThreads = std::thread::hardware_concurrency();

		const size_t size = last - first;

		if (size < 2)
			return;

		// No parallelize small arrays
		if (size < 32) {
			std::exclusive_scan(first, last, ofirst, type());
			return;
		}

		const std::vector<size_t> ranges = computeRanges(size, nThreads);

		std::vector<std::thread> threads(nThreads);

		std::atomic<size_t> barrier1(nThreads);
		std::atomic<size_t> barrier2(1);

		std::vector<size_t> count(nThreads);

		std::mutex mtx;

		// First scan by chunks
		for (size_t i = 0; i < nThreads; ++i) {
			threads[i] = std::thread(
				[&, i]() {
					std::exclusive_scan(
						first + ranges[i],
						first + ranges[i + 1],
						ofirst + ranges[i],
						type()
					);

					// Busy waiting (just for fun)
					if (barrier1.load() == 0)
						abort();
					barrier1.fetch_sub(1);
					while (barrier1.load() > 0)
						std::this_thread::yield();

					if (i == 0) {
						// We assume that the number of threads is small enough
						// at least nThreads << size
						for (size_t j = 1; j < nThreads; ++j) {
							const size_t lastIdx = ranges[j] - 1;
							count[j] = count[j - 1] + *(ofirst + lastIdx) + *(first + lastIdx);
						}
						barrier2.fetch_sub(1);
						barrier2.notify_all();
					}

					// light wait waiting
					barrier2.wait(1);
					std::cout << "Thread: " << i << "after" << std::endl;

					// Update indices
					std::for_each(
						ofirst + ranges[i],
						ofirst + ranges[i + 1],
						[value = count[i]](auto& v)
						{
							v += value;
						}
					);
				}
			);
		}

		for (auto &thread : threads)
			thread.join();
	}



	
}
