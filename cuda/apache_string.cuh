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
#include <fstream>
#include <cassert>

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
__global__ void setStarts(
	const size_t *tmp, size_t size, size_t *starts, size_t nWords
) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		if (idx == 0 || tmp[idx] != tmp[idx - 1]) {
			assert(tmp[idx] < nWords);
			starts[tmp[idx]] = idx;
		}
	}

	if (idx == size - 1)
		starts[tmp[size - 1] + 1] = size;
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

	if (idx == nWords - 1)
		buffer[size - nWords] = '\0';
}

template<int blockdim>
void parseString(const std::string &input)
{
	const size_t size = input.size();

	char *d_in;
	cudaMalloc((void**)&d_in, size * sizeof(char));
    cudaMemcpy(d_in, input.data(), input.size() * sizeof(char), cudaMemcpyHostToDevice);

	size_t* tmp;
	cudaMalloc((void**)&tmp, size * sizeof(size_t));

	int numBlocks = (size + blockdim - 1) / blockdim;

	markSpaces<<<numBlocks, blockdim>>>(d_in, size, tmp);
	scan_internal<blockdim, size_t>(tmp, size, tmp);

	size_t nWords;
	cudaMemcpy(&nWords, &tmp[size], sizeof(size_t), cudaMemcpyDeviceToHost);

	size_t *starts;
	cudaMalloc((void**)&starts, (nWords + 1) * sizeof(size_t));
	setStarts<<<numBlocks, blockdim>>>(tmp, size, starts, nWords);

	// Reuse the tmp buffer to avoid an extra malloc and free
	// Update the number of blocks for now on
	numBlocks = (nWords + blockdim - 1) / nWords;
	char *buffer = (char *)tmp;
	transformBuffer<<<numBlocks, blockdim>>>(d_in, size, starts, nWords, buffer, size - nWords);

	std::string h_buffer(size - nWords, ' ');
	std::vector<size_t> h_starts(nWords + 1);

	cudaMemcpy(h_buffer.data(), buffer, (size - nWords) * sizeof(char), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_starts.data(), starts, (nWords + 1) * sizeof(size_t), cudaMemcpyDeviceToHost);

	cudaFree(starts);
	cudaFree(tmp);
	cudaFree(d_in);
}


// __global__ void sort(char *input, int *offsets, int *sizes, int &numStrings)
// {
// 	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (idx == 0 || input[idx - 1] == ' ') {

// 		int length = 0;
// 		while (input[idx + length] != ' ' && input[idx + length] != '\0')
// 			length++;

// 		int wordIdx = atomicAdd(numStrings, 1);
// 		offsets[wordIdx] = idx;
// 		sizes[wordIdx] = length;
// 	}
// }


/**
   Perform a boolean scan
 */

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
	}

	friend std::ostream &operator <<(std::ostream &out, const apache_string &str)
	{
		for (auto it = str.starts.begin(); std::next(it) != str.starts.end(); std::advance(it, 1))
		{
			out << str.buffer.substr(*it, *std::next(it) - *it) << " ";
		}
		return out;
	}

};
