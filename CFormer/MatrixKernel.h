#pragma once
#include "Utils.h"

class MatrixKernel {

public:

	template<typename Function, typename... Params>
	static void runElementKernel(int height, int width, int sharedMemory, Function function, Params... params) {
		int N = height * width;
		int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
		if (sharedMemory == 0) {
			function << < numBlocks, Utils::THREADS_PER_BLOCK >> > (forward<Params>(params)...);
		}
		else {
			function << < numBlocks, Utils::THREADS_PER_BLOCK, sharedMemory >> > (forward<Params>(params)...);
		}
	}
	template<typename Function, typename... Params>
	static void runRowKernel(int height, int width, int sharedMemory, Function function, Params... params) {
		if (sharedMemory == 0) {
			function << < height, Utils::THREADS_PER_BLOCK >> > (forward<Params>(params)...);
		}
		else {
			function << < height, Utils::THREADS_PER_BLOCK, sharedMemory >> > (forward<Params>(params)...);
		}
	}
	template<typename Function, typename... Params>
	static void runColumnKernel(int height, int width, int sharedMemory, Function function, Params... params) {
		if (sharedMemory == 0) {
			function << < width, Utils::THREADS_PER_BLOCK >> > (forward<Params>(params)...);
		}
		else {
			function << < width, Utils::THREADS_PER_BLOCK, sharedMemory >> > (forward<Params>(params)...);
		}
	}
	template<typename Function, typename... Params>
	static void runElementKernelBatched(int batchSize, int height, int width, int sharedMemory, Function function, Params... params) {
		int N = height * width;
		int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
		dim3 blocks(numBlocks, batchSize);
		if (sharedMemory == 0) {
			function << < blocks, Utils::THREADS_PER_BLOCK >> > (forward<Params>(params)...);
		}
		else {
			function << < blocks, Utils::THREADS_PER_BLOCK, sharedMemory >> > (forward<Params>(params)...);
		}
	}
	template<typename Function, typename... Params>
	static void runRowKernelBatched(int batchSize, int height, int width, int sharedMemory, Function function, Params... params) {
		dim3 blocks(height, batchSize);
		if (sharedMemory == 0) {
			function << < blocks, Utils::THREADS_PER_BLOCK >> > (forward<Params>(params)...);
		}
		else {
			function << < blocks, Utils::THREADS_PER_BLOCK, sharedMemory >> > (forward<Params>(params)...);
		}
	}
	template<typename Function, typename... Params>
	static void runColumnKernelBatched(int batchSize, int height, int width, int sharedMemory, Function function, Params... params) {
		dim3 blocks(width, batchSize);
		if (sharedMemory == 0) {
			function << < blocks, Utils::THREADS_PER_BLOCK >> > (forward<Params>(params)...);
		}
		else {
			function << < blocks, Utils::THREADS_PER_BLOCK, sharedMemory >> > (forward<Params>(params)...);
		}
	}
};

