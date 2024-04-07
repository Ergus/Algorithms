
#include <stdio.h>

int main() {

	int nDevices;
	cudaGetDeviceCount(&nDevices);

	printf("Number of devices: %d\n", nDevices);

	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);

		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);

		printf("  Compute Capability minor-major: %d-%d\n", prop.minor, prop.major);
		printf("  Warp-size: %d\n", prop.warpSize);

		printf("\n  ===================================================\n");
		printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate/1024);
		printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %.1f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
		printf("  minor-major: %d-%d\n", prop.minor, prop.major);
		printf("  Warp-size: %d\n", prop.warpSize);

		printf("\n  ===================================================\n");
		printf("  Total Global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
		printf("  Total Const memory (Kbytes) %.1f\n",(float)(prop.totalConstMem)/1024.0);
		printf("  l2CacheSize (Mbytes) %.1f\n", (float)(prop.l2CacheSize)/1024.0/1024.0);

		printf("\n  ===================================================\n");
		printf("  Max threads per block %d\n", prop.maxThreadsPerBlock);
		printf("  32-bits registers per block %d\n", prop.regsPerBlock);
		printf("  Shared memory per block (Kbytes) %d\n", (float)(prop.sharedMemPerBlock)/1024.0);

		printf("\n  ===================================================\n");
		printf("  Number of multi processors %d\n", prop.multiProcessorCount);
		printf("  Max threads per multi processor %d\n", prop.maxThreadsPerMultiProcessor);
		printf("  Max blocks per multi processor %d\n", prop.maxBlocksPerMultiProcessor);

		printf("\n  ===================================================\n");
		printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
		printf("  Cooperative Launch: %s\n", prop.cooperativeLaunch ? "yes" : "no");
		printf("  Concurrent computation/communication: %s\n",prop.deviceOverlap ? "yes" : "no");
		printf("  Stream priorities supported: %s\n", prop.streamPrioritiesSupported ? "yes" : "no");
		printf("  Unified Addressing: %s\n", prop.unifiedAddressing ? "yes" : "no");
	}

	return 0;
}
