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

#include <vector>
#include <string>
#include <cassert>
#include <algorithm>

#include "scan.cuh"

/**
   Create a temporal array with 0 everywhere and 1 in the spaces positions
   "aaaa bbbbb ccc" -> "00001000001000"
*/
__global__ void mark_spaces(const char *input, size_t size, size_t *tmp)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		tmp[idx] = (input[idx] == ' ');
}

/**
   Fill the starts arrays with the indices of words starts
   "000011111122223" -> "0,4,10,14"
 **/
__global__ void set_starts_in(
	const size_t *tmp, size_t size, size_t *starts, size_t nWords
) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		if (idx == 0 || tmp[idx] != tmp[idx - 1]) {
			starts[tmp[idx]] = idx;
		}
	}

	if (idx == nWords - 1)
		starts[nWords] = size + 1;
}

/**
   Compact the string using the start indices.
   "aaaa bbb ccc" -> "aaabbbccc"
 **/
__global__ void transform_buffer(
	const char *input, size_t size,
	const size_t *starts, size_t nWords,
	char *buffer, size_t buffersize
) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nWords) {
		size_t inStart = starts[idx];
		size_t inEnd = starts[idx + 1];
		for (size_t i = inStart; i < inEnd - 1; ++i)
			buffer[i - idx] = input[i];
	}
}

/**
   The starts array contains the indices of the start points in the original array.
   This functions updates the indices to the ones in the final buffer.
 **/
__global__ void update_starts(size_t *starts, size_t nWords)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx <= nWords)
		starts[idx] -= idx;
}

/**
   This is the main cuda function.

   This gets a string and returns a pair with a string with no spaces
   and a vector with the string start positions.
   I assume that the input strings are trimmed and this version of the code
   works totally reentrant and uses the default stream (yes I know that's
   not efficient, but This was made for correctness mainly)
 **/
template<int blockdim>
std::pair<std::string, std::vector<size_t>> parse_string_gpu(const std::string &input)
{
	const size_t size = input.size();

	char *d_in;
	cudaMalloc((void**)&d_in, size * sizeof(char));
    cudaMemcpy(d_in, input.data(), input.size() * sizeof(char), cudaMemcpyHostToDevice);

	size_t* ones;
	cudaMalloc((void**)&ones, size * sizeof(size_t));

	int numBlocks = (size + blockdim - 1) / blockdim;

	mark_spaces<<<numBlocks, blockdim>>>(d_in, size, ones);
	scan_internal<blockdim, size_t>(ones, size, ones);

#ifndef NDEBUG
	{
		size_t *hones = (size_t *) malloc(size * sizeof(size_t));

		cudaMemcpy(hones, ones, size * sizeof(size_t), cudaMemcpyDeviceToHost);

		size_t count = 0;
		for (size_t i = 1; i < size; ++i) {
			if (hones[i] != hones[i - 1]) {
				myassert(input[i - 1] == ' ');
				++count;
			} else if (input[i - 1] == ' ') {
				perror("Some space was not marked in gpu");
			}
		}
		myassert(count == std::count_if(input.begin(), input.end(), [](char c) { return (c == ' '); }));
		myassert(count == hones[size - 1]);

		free(hones);
	}
#endif

	size_t nWords;
	cudaMemcpy(&nWords, &ones[size - 1], sizeof(size_t), cudaMemcpyDeviceToHost);
	nWords += 1; // Words are spaces + 1, because the first letter is not preceded by a space.

	size_t *starts;
	cudaMalloc((void**)&starts, (nWords + 1) * sizeof(size_t));
	set_starts_in<<<numBlocks, blockdim>>>(ones, size, starts, nWords);

#ifndef NDEBUG
	{
		size_t *hstarts = (size_t *) malloc((nWords + 1) * sizeof(size_t));
		cudaMemcpy(hstarts, starts, (nWords + 1) * sizeof(size_t), cudaMemcpyDeviceToHost);

		//myassert(hstarts[0] == 0);
		for (size_t i = 1; i < nWords; ++i) {
			size_t idx = hstarts[i];
			myassert(input[idx - 1] == ' ');
		}

		free(hstarts);
	}
#endif

	// Reuse the ones buffer to avoid an extra malloc and free
	// Update the number of blocks for now on
	numBlocks = (nWords + blockdim - 1) / blockdim;
	char *buffer = (char *)ones;
	transform_buffer<<<numBlocks, blockdim>>>(d_in, size, starts, nWords, buffer, size - nWords);
	update_starts<<<numBlocks, blockdim>>>(starts, nWords);

	char *hbuffer = (char *)malloc((size - nWords + 1) * sizeof(char));

	cudaMemcpy(hbuffer, buffer, (size - nWords + 1) * sizeof(char), cudaMemcpyDeviceToHost);

	std::string h_buffer(hbuffer, (size - nWords + 1));

	std::vector<size_t> h_starts(nWords + 1);

	cudaMemcpy(h_starts.data(), starts, (nWords + 1) * sizeof(size_t), cudaMemcpyDeviceToHost);

	free(hbuffer);
	cudaFree(starts);
	cudaFree(ones);
	cudaFree(d_in);

	return {h_buffer, h_starts};
}

std::pair<std::string, std::vector<size_t>> parse_string_cpu(const std::string &input)
{
	std::string buffer;         //! Final buffer with no spaces
	std::vector<size_t> starts; //! Array of start indices + count.

	starts.push_back(0);
	for(char c : input) {
		if (c == ' ')
			starts.push_back(buffer.size());
		else
			buffer.push_back(c);
	}
	starts.push_back(buffer.size());

	size_t nWords = starts.size() - 1;

	return {buffer, starts};
}

/**
    Apache class helper just to simplify the user tests code.

    This class has two constructors one uses cpu scan and the other
    uses gpu.
    It also implements the == operator for testing purposes.
 **/
class apache_string {
	std::string buffer;         //! Final buffer with no spaces
	std::vector<size_t> starts; //! Array of start indices + count.

	apache_string() = default;

public:
	template<int I = 0>
	static apache_string factory(const std::string &input)
	{
		auto val = apache_string();
		if constexpr (I == 0) {
			auto [_buffer, _starts] = parse_string_cpu(input);
			val.buffer = std::move(_buffer);
			val.starts = std::move(_starts);
		} else {
			auto [_buffer, _starts] = parse_string_gpu<I>(input);
			val.buffer = std::move(_buffer);
			val.starts = std::move(_starts);
		}
		return val;
	}

	friend std::ostream &operator <<(std::ostream &out, const apache_string &str)
	{
		for (auto it = str.starts.begin(); std::next(it) != str.starts.end(); std::advance(it, 1))
		{
			out << str.buffer.substr(*it, *std::next(it) - *it) << " ";
		}
		return out;
	}

	bool operator==(const apache_string& other) const
	{
		return (buffer == other.buffer) && (starts == other.starts);
	}

};
