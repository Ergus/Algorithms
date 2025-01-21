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
__global__ void markSpaces(const char *input, size_t size, size_t *tmp)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		tmp[idx] = (input[idx] == ' ');
}

/**
   Fill the starts arrays with the indices of words starts
   "000011111122223" -> "0,4,10,14"
 */
__global__ void setStartsIn(
	const size_t *tmp, size_t size, size_t *starts, size_t nWords
) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		if (idx == 0 || tmp[idx] != tmp[idx - 1]) {
			starts[tmp[idx]] = idx;
		}
	}

	if (idx == size)
		starts[nWords] = idx + 1;
}



__global__ void transformBuffer(
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

__global__ void updateStarts(size_t *starts, size_t nWords)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx <= nWords)
		starts[idx] -= idx;
}

template<int blockdim>
std::pair<std::string, std::vector<size_t>> parseString(const std::string &input)
{
	const size_t size = input.size();

	char *d_in;
	cudaMalloc((void**)&d_in, size * sizeof(char));
    cudaMemcpy(d_in, input.data(), input.size() * sizeof(char), cudaMemcpyHostToDevice);

	size_t* ones;
	cudaMalloc((void**)&ones, size * sizeof(size_t));

	int numBlocks = (size + blockdim - 1) / blockdim;

	markSpaces<<<numBlocks, blockdim>>>(d_in, size, ones);
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
	setStartsIn<<<numBlocks, blockdim>>>(ones, size, starts, nWords);

#ifndef NDEBUG
	{
		size_t *hstarts = (size_t *) malloc((nWords + 1) * sizeof(size_t));
		cudaMemcpy(hstarts, starts, (nWords + 1) * sizeof(size_t), cudaMemcpyDeviceToHost);

		myassert(hstarts[0] == 0);
		for (size_t i = 1; i <= nWords; ++i) {
			size_t idx = hstarts[i];
			myassert(input[idx] != ' ');
		}

		free(hstarts);
	}
#endif

	// Reuse the ones buffer to avoid an extra malloc and free
	// Update the number of blocks for now on
	numBlocks = (nWords + blockdim - 1) / blockdim;
	char *buffer = (char *)ones;
	transformBuffer<<<numBlocks, blockdim>>>(d_in, size, starts, nWords, buffer, size - nWords);
	updateStarts<<<numBlocks, blockdim>>>(starts, nWords);

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

class apache_string {
	std::string buffer;
	std::vector<size_t> starts;

public:
	apache_string(const std::string &input)
	{
		starts.push_back(0);
		for(char c : input) {
			if (c == ' ')
				starts.push_back(buffer.size());
			else
				buffer.push_back(c);
		}
		starts.push_back(buffer.size());

		size_t nWords = starts.size() - 1;
	}

	apache_string(const std::string &input, size_t blockdim)
	{
		auto [_buffer, _starts] = parseString<32>(input);
		buffer = std::move(_buffer);
		starts = std::move(_starts);
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
