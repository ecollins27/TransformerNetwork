#pragma once

class MatrixKernel {

public:
	static const int THREADS_PER_BLOCK = 256;

	template<typename Function, typename... Params>
	static void runElementKernel(int height, int width, int sharedMemory, Function function, Params... params) {
		int N = height * width;
		int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		if (sharedMemory == 0) {
			function << < numBlocks, THREADS_PER_BLOCK >> > (forward<Params>(params)...);
		}
		else {
			function << < numBlocks, THREADS_PER_BLOCK, sharedMemory >> > (forward<Params>(params)...);
		}
	}
	template<typename Function, typename... Params>
	static void runRowKernel(int height, int width, int sharedMemory, Function function, Params... params) {
		if (sharedMemory == 0) {
			function << < height, THREADS_PER_BLOCK >> > (forward<Params>(params)...);
		}
		else {
			function << < height, THREADS_PER_BLOCK, sharedMemory >> > (forward<Params>(params)...);
		}
	}
	template<typename Function, typename... Params>
	static void runColumnKernel(int height, int width, int sharedMemory, Function function, Params... params) {
		if (sharedMemory == 0) {
			function << < width, THREADS_PER_BLOCK >> > (forward<Params>(params)...);
		}
		else {
			function << < width, THREADS_PER_BLOCK, sharedMemory >> > (forward<Params>(params)...);
		}
	}
};

