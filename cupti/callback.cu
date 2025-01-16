// Copyright (C) 2024  Jimmy Aguilar Mena

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.	 If not, see <http://www.gnu.org/licenses/>.

#pragma nv_diag_suppress 815 // suppress consr return warning
#include <argparser.hpp>

#include <vector>
#include <random>
#include <algorithm>

#include "utils.h"


#include <cupti.h>


/**
   Example kernel r[] = a1[] + a2[]
*/
__global__ void kernelExample(size_t size, const float *a1, const float *a2, float *r)
{
	int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (globalIdx < size)
		r[globalIdx] = a1[globalIdx] + a2[globalIdx];
}


/**
   Wrapper function for kernel r[] = a1[] + a2[]
*/
std::vector<float> functionExample(
	const std::vector<float> &a1,
	const std::vector<float> &a2
) {
	const size_t size = a1.size();
	myassert(a2.size() ==  size);

	float *d_a1, *d_a2, *d_r;
	cudaMalloc((void**)&d_a1, size * sizeof(float));
	cudaMalloc((void**)&d_a2, size * sizeof(float));
	cudaMalloc((void**)&d_r, size * sizeof(float));

	cudaMemcpy(d_a1, a1.data(), size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_a2, a2.data(), size * sizeof(float), cudaMemcpyHostToDevice);

	constexpr size_t blockdim = 128;
	size_t nblocks = (size + blockdim - 1) / blockdim;

	kernelExample<<<nblocks, blockdim>>>(size, d_a1, d_a2, d_r);

	cudaFree(d_a2);
	cudaFree(d_a1);

	std::vector<float> r(size);
	cudaMemcpy(r.data(), &d_r[0], size * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_r);

	return r;
}


static void CUPTIAPI My_CUPTI_callback(
	void* /* udata */,
	CUpti_CallbackDomain domain,
	CUpti_CallbackId cbid,
	const CUpti_CallbackData *cbinfo
) {
	if (cbinfo == NULL)
		return;

	/*
	 * Filter out all nested callbacks
	 */
	switch (cbinfo->callbackSite) {
	case CUPTI_API_ENTER:
		printf("Callback Enter %d\n", domain);
		break;
	case CUPTI_API_EXIT:
		printf("Callback Exit %d\n", domain);
		break;
	}

	// switch (domain) {
	// case CUPTI_CB_DOMAIN_DRIVER_API:
	// 	ret = Extrae_DriverAPI_callback(cbid, cbinfo);
	// 	break;
	// case CUPTI_CB_DOMAIN_RUNTIME_API:
	// 	ret = Extrae_RuntimeAPI_callback(cbid, cbinfo);
	// 	break;
	// default:
	// 	ret = -1;
	// }

	// if (ret == 0) {
	// 	if (cbinfo->callbackSite == CUPTI_API_ENTER)
	// 		TRACE_EVENT(LAST_READ_TIME, CUDA_UNTRACKED_EV, cbid);
	// 	else if (cbinfo->callbackSite == CUPTI_API_EXIT)
	// 		TRACE_EVENT(TIME, CUDA_UNTRACKED_EV, EVT_END);
	// }

	// if((cbinfo->callbackSite == CUPTI_API_EXIT))
	// 	Extrae_CUDA_updateDepth(cbinfo);
}

void My_CUPTI_init()
{
	CUpti_SubscriberHandle subscriber;

	/* Create a subscriber. All the routines will be handled at Extrae_CUPTI_callback */
	cuptiSubscribe(&subscriber, (CUpti_CallbackFunc) My_CUPTI_callback, NULL);

	/* Enable all callbacks in all domains */
	cuptiEnableAllDomains(1, subscriber);

	/* Disable uninteresting callbacks */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaDriverGetVersion_v3020); /* 1 */
}

int main(int argc, char **argv)
{
	My_CUPTI_init();

	argparser::init(argc, argv);
	const size_t size = argparser::cl<int>("array_size");

	std::vector<float> a1(size);
	std::vector<float> a2(size);

	static std::random_device rd;	 // you only need to initialize it once
	static std::mt19937 mte(rd());	 // this is a relative big object to create

	std::uniform_int_distribution<int> dist(0, 1024); // dist(mte)

	std::generate(a1.begin(), a1.end(), [&dist](){ return dist(mte); });
	std::generate(a2.begin(), a2.end(), [&dist](){ return dist(mte); });

	std::vector<float> result = functionExample(a1, a2);

	// In this case the float sum in gpu is exactly the same.
	for (size_t i = 0; i < result.size(); ++i)
		myassert(a1[i] + a2[i] == result[i]);

	return 0;
}
