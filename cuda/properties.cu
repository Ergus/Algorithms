
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

		const int* mgs = prop.maxGridSize;
		printf("  Max Grid Size: [%d,%d;%d]\n", mgs[0], mgs[1], mgs[2]);

		printf("\n  ===================================================\n");
		printf("  Total Global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
		printf("  Total Const memory (Kbytes) %.1f\n",(float)(prop.totalConstMem)/1024.0);
		printf("  Total l2 Cache Size (Mbytes) %.1f\n", (float)(prop.l2CacheSize)/1024.0/1024.0);
		printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate/1024);
		printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %.1f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

		printf("\n  ===================================================\n");
		printf("  Max threads per block %d\n", prop.maxThreadsPerBlock);
		printf("  32-bits registers per block %d\n", prop.regsPerBlock);
		printf("  Shared memory per block (Kbytes) %.1f\n", (float)(prop.sharedMemPerBlock)/1024.0);

		printf("\n  ===================================================\n");
		printf("  Number of SM %d\n", prop.multiProcessorCount);
		printf("  Max threads per SM %d\n", prop.maxThreadsPerMultiProcessor);
		printf("  Max blocks per SM %d\n", prop.maxBlocksPerMultiProcessor);
		printf("  Shared memory per SM (Kbytes) %.1f\n", (float)(prop.sharedMemPerMultiprocessor)/1024.0);
		printf("  Number of registers per SM %d\n", prop.regsPerMultiprocessor);

		printf("\n  ===================================================\n");
		printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
		printf("  Cooperative Launch: %s\n", prop.cooperativeLaunch ? "yes" : "no");
		printf("  Concurrent computation/communication: %s\n",prop.deviceOverlap ? "yes" : "no");
		printf("  Stream priorities supported: %s\n", prop.streamPrioritiesSupported ? "yes" : "no");
		printf("  Unified Addressing: %s\n", prop.unifiedAddressing ? "yes" : "no");
		printf("  Support Managed Memory: %d\n", prop.managedMemory ? "yes" : "no");
	}

	return 0;
}
