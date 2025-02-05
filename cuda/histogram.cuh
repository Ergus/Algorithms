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

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "utils.cuh"
#include <iostream>

template<typename T>
__device__ T atomicMax(T *addr, T val);

template<typename T>
__device__ T atomicMin(T *addr, T val);

template <>
__device__ double atomicMax<double>(double *addr, double val) {
    unsigned long long *addr_as_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_as_ull, assumed;

    do {
        assumed = old;
        double d_old = __longlong_as_double(old);
        if (d_old >= val) break; // Already the min value
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);

    return __longlong_as_double(old);
}

template <>
__device__ double atomicMin<double>(double *addr, double val) {
    unsigned long long *addr_as_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_as_ull, assumed;

    do {
        assumed = old;
        double d_old = __longlong_as_double(old);
        if (d_old <= val) break; // Already the min value
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);

    return __longlong_as_double(old);
}

template <typename T, typename Tbins>
struct histogram {
	T hmin, hmax;
	size_t nbins;
	Tbins bins; // Allocated as a payload

	T getBinSize() const
	{
		return (hmax - hmin) / nbins;
	}

	std::pair<T, unsigned int> operator[](size_t i) const
	{
		const T binsize = getBinSize();
		return {hmin + (i * binsize),  bins[i]};
	}

	friend std::ostream& operator<<(std::ostream &out, const histogram &hist)
	{
		const T binsize = hist.getBinSize();

		out << "Hist range: [" << hist.hmin << ":" << hist.hmax << "]" << std::endl;
		out << "N Bins: " << hist.nbins << " Bin size: " << binsize << std::endl;
		for (size_t i = 0; i < hist.nbins; ++i) {
			std::cout << "{" << hist.hmin + (i * binsize) << " : " << hist.bins[i] << "}\n";
		}

		return out;
	}

	template <typename Obins>
	bool operator==(const histogram<T, Obins> other) const
	{
		for (size_t i = 0; i < nbins; ++i) {
			if (bins[i] != other.bins[i]) {
				std::cerr << (*this)[i] << "!=" << other[i] << std::endl;
				return false;
			}
		}

		return hmin == other.hmin
			&& hmax == other.hmax
			&& nbins == other.nbins;

	}
};

template<typename T>
using hist_ptr = histogram<T, unsigned int *>;

template<typename T>
using hist_vec = histogram<T, std::vector<unsigned int>>;

template<typename T>
__global__ void get_min_max(const T *data, const size_t size, hist_ptr<T>* out)
{
	cg::grid_group grid = cg::this_grid();
	cg::thread_block cta = cg::this_thread_block();
	cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());

	const int lane = warp.thread_rank();
	const int wid = warp.meta_group_rank();
	const int nwarps = warp.meta_group_size();

	assert(grid.is_valid());

	__shared__ T sharedMin[32]; // must have size blockdim
	__shared__ T sharedMax[32];

	T lmin = INFINITY;
	T lmax = -INFINITY;

	for (int i = grid.thread_rank() ; i < size; i += grid.num_threads()) {

		T tmp = data[i];
		lmin = min(lmin, tmp);
		lmax = max(lmax, tmp);
	}

	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) {
		lmin = min(lmin, warp.shfl_down(lmin, offset));
		lmax = max(lmax, warp.shfl_down(lmax, offset));
	}

	// All lanes 0 write the variable to shared memory
	if (lane == 0) {
		sharedMin[wid] = lmin;
		sharedMax[wid] = lmax;
	}

	__syncthreads();

	// If in warp 0 the perform local reduction
	if (wid == 0) {

		lmin = (lane < nwarps) ? sharedMin[lane] : INFINITY;
		lmax = (lane < nwarps) ? sharedMax[lane] : -INFINITY;

		#pragma unroll
		for (int offset = 16; offset > 0; offset >>= 1) {
			lmin = min(lmin, warp.shfl_down(lmin, offset));
			lmax = max(lmax, warp.shfl_down(lmax, offset));
		}

		if (lane == 0) {
			atomicMin<T>(&out->hmin, lmin);
			atomicMax<T>(&out->hmax, lmax);
		}
	}
}

template<typename T>
__global__ void build_histogram(const T *in, size_t size, hist_ptr<T> *hist)
{
	cg::grid_group grid = cg::this_grid();
	cg::thread_block cta = cg::this_thread_block();

	hist_ptr lhist = *hist;
	const T binsize = (lhist.hmax - lhist.hmin) / lhist.nbins;

	// Initialize the output to zero.
	extern __shared__ uint sharedData[];

	for (size_t idx = grid.thread_rank(); idx < lhist.nbins; idx += grid.num_threads())
		lhist.bins[idx] = 0;

	for (size_t idx = cta.thread_rank(); idx < lhist.nbins; idx += cta.num_threads())
		sharedData[idx] = 0;

	__syncthreads();


	// Now run over the whole array
	for (int idx = grid.thread_rank(); idx < size; idx += grid.num_threads()) {
		unsigned int bin = (in[idx] - lhist.hmin) / binsize;
		atomicAdd(&sharedData[bin], 1);
	}

	__syncthreads();

	for (int idx = cta.thread_rank(); idx < lhist.nbins; idx += cta.num_threads()) {
		atomicAdd(&lhist.bins[idx], sharedData[idx]);
	}
}

template <typename T>
hist_vec<T> histogram_gpu(const std::vector<T> input, size_t nbins)
{
	// Events to measure time
	cudaErrorCheck err;
	cudaEvent_t eStart, eStop;
	cudaEventCreate(&eStart);
	cudaEventCreate(&eStop);

	// Compute max concurrent occupancy combination
	// Compute blockSize; The grid size from here may not be accurate enough because
	// we need to use the sharedSize to get the accurate value for numBlocksPerSm

	hist_ptr<T> *d_hist;
	T *d_input;
	cudaEventRecord(eStart);
	err = cudaMalloc((void**)&d_input, input.size() * sizeof(T));

	unsigned int *d_bins;
	err = cudaMalloc((void**)&d_hist, sizeof(hist_ptr<T>));
	err = cudaMalloc((void**)&d_bins, nbins * sizeof(unsigned int));
	cudaEventRecord(eStop);
	fprintf(stderr, "# cudaMalloc time %f mS\n", myGetElapsed(eStart, eStop));

	cudaEventRecord(eStart);

	size_t size = input.size();
	hist_ptr<T> h_hist = {
		.hmin = std::numeric_limits<T>::max(),
		.hmax = std::numeric_limits<T>::min(),
		.nbins = nbins, // Allocated as a payload
		.bins = d_bins
	};

	cudaMemcpy(d_input, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_hist, &h_hist, sizeof(hist_ptr<T>), cudaMemcpyHostToDevice);
	cudaEventRecord(eStop);
	fprintf(stderr, "# cudaMemcpy h2d time %f mS\n", myGetElapsed(eStart, eStop));

	cudaEventRecord(eStart);
	auto [dim_grid1, dim_block1] = get_grid_block_dims(
		size,
		get_min_max<T>,
		0,
		size
	);
	void* kernel1Args[] = { &d_input, &size, &d_hist };
	err = cudaLaunchCooperativeKernel(
		(void *) get_min_max<T>,
		dim_grid1,
		dim_block1,
		kernel1Args
	);
	cudaEventRecord(eStop);
	fprintf(stderr, "# Kernel get_min_max time %f mS\n", myGetElapsed(eStart, eStop));

	cudaEventRecord(eStart);
	auto [dim_grid2, dim_block2] = get_grid_block_dims(
		size,
		build_histogram<T>,
		nbins * sizeof(unsigned int),
		size
	);
	void* kernel2Args[] = { &d_input, &size, &d_hist };
	err = cudaLaunchCooperativeKernel(
		(void *) build_histogram<T>,
		dim_grid2,
		dim_block2,
		kernel2Args,
		nbins * sizeof(unsigned int)
	);
	cudaEventRecord(eStop);

	cudaDeviceSynchronize();
	fprintf(stderr, "# Kernel build_histogram time %f mS\n", myGetElapsed(eStart, eStop));

	cudaEventRecord(eStart);

	std::vector<unsigned int> tmp(nbins);

	err = cudaMemcpy(&h_hist, d_hist, sizeof(hist_ptr<T>), cudaMemcpyDeviceToHost);
	err = cudaMemcpy(tmp.data(), d_bins, nbins * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	hist_vec<T> ret {
		.hmin = h_hist.hmin,
		.hmax = h_hist.hmax,
		.nbins = h_hist.nbins,
		.bins = std::move(tmp)
	};

	cudaEventRecord(eStop);
	fprintf(stderr, "# cudaMemcpy d2h  time %f mS\n", myGetElapsed(eStart, eStop));

	cudaFree(d_hist);
	cudaFree(d_bins);
	cudaFree(d_input);

	return ret;
}

template <typename T>
hist_vec<T> histogram_cpu(const std::vector<T> input, size_t nbins)
{
	auto [lmin, lmax] = std::minmax_element(input.begin(), input.end());

	std::vector<unsigned int> bins(nbins);

	const T binsize = (*lmax - *lmin) / nbins;

	hist_vec<T> ret {
		.hmin = *lmin,
		.hmax = *lmax,
		.nbins = nbins,
		.bins = std::vector<unsigned int>(nbins)
	};

	for (const T &val : input) {
		unsigned int bin = (val - ret.hmin) / binsize;
		++ret.bins[bin];
	}

	return ret;
}
