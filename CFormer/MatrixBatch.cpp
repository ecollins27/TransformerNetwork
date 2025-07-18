#include "MatrixBatch.h"
#include "Matrix2.h"

float MatrixBatch::ALPHA = 1.0f;
float MatrixBatch::BETA0 = 0.0f;
float MatrixBatch::BETA1 = 1.0f;
int MatrixBatch::NUM_DEVICES = 0;
int* MatrixBatch::DEVICE_BATCHSIZES = NULL;
int* MatrixBatch::DEVICE_LENGTHS = NULL;
float**** MatrixBatch::DEVICES = NULL;
float**** MatrixBatch::HOST_DEVICES = NULL;

MatrixBatch::MatrixBatch(int batchSize, int height, int width, int threadNum) {
	maxLength = length;
	this->length = maxLength;
	this->batchSize = batchSize;
	this->height = height;
	this->width = width;
	this->threadNum = threadNum;
	cudaError_t err = cudaMallocHost(&host, batchSize * sizeof(float*));
	for (int i = 0; i < batchSize; i++) {
		err = cudaMallocHost(&host[i], maxLength * sizeof(float));
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
		}
	}
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < length; j++) {
			host[i][j] = 0;
		}
	}
}

MatrixBatch::MatrixBatch(FillFunction& fillFunction, int batchSize, int height, int width, int threadNum) {
	maxLength = height * width;
	this->length = maxLength;
	this->batchSize = batchSize;
	this->height = height;
	this->width = width;
	this->threadNum = threadNum;
	cudaError_t err = cudaMallocHost(&host, batchSize * sizeof(float*));
	for (int i = 0; i < batchSize; i++) {
		err = cudaMallocHost(&host[i], maxLength * sizeof(float));
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
		}
	}
	fill(fillFunction);
}

void MatrixBatch::free() {
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaFreeHost(host[i]);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory deallocation failed: ") + cudaGetErrorString(err));
		}
	}
	err = cudaFreeHost(host);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory deallocation failed: ") + cudaGetErrorString(err));
	}
}

int MatrixBatch::e(int i, int j) {
	return i + height * j;
}

float& MatrixBatch::operator()(int i, int j, int k) {
	return host[i][e(j, k)];
}

void MatrixBatch::fill(FillFunction& fillFunction) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < width; k++) {
				host[i][e(j, k)] = fillFunction(j, k);
			}
		}
	}
}

__global__
void kernelConstantFill(float c, float** A, int N) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;

	if (n < N) {
		A[batch][n] = c;
	}
}

void MatrixBatch::constantFill(float c) {
	int N = length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1);
	dim3 blocks(numBlocks, batchSize);
	kernelConstantFill <<< blocks, Utils::THREADS_PER_BLOCK >>> (c, DEVICES[threadNum][0], N);
	copyToHost(0);
}

__global__
void kernelScale(float c, float** A, int N) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;

	if (n < N) {
		A[batch][n] *= c;
	}
}

void MatrixBatch::scale(float c) {
	int N = length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1);
	dim3 blocks(numBlocks, batchSize);
	copyToDevice(0);
	kernelScale << < blocks, Utils::THREADS_PER_BLOCK >> > (c, DEVICES[threadNum][0], N);
	copyToHost(0);
}

__global__
void kernelSqrt(float** A, float** B, int N) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;

	if (n < N) {
		B[batch][n] = sqrt(A[batch][n]);
	}
}

void MatrixBatch::sqrt(MatrixBatch& B) {
	int N = length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1);
	dim3 blocks(numBlocks, batchSize);
	copyToDevice(0);
	kernelSqrt <<< blocks, Utils::THREADS_PER_BLOCK >> > (DEVICES[threadNum][0], DEVICES[threadNum][1], N);
	B.copyToHost(1, N);
}

__global__
void kernelCondense(float** matrix, float* condensed, int batchSize, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float sum = 0;
		for (int b = 0; b < batchSize; b++) {
			sum += matrix[b][i];
		}
		condensed[i] = sum;
	}
}

void MatrixBatch::condense(Matrix2& B) {
	int N = length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	copyToDevice(0);
	kernelCondense <<< numBlocks, Utils::THREADS_PER_BLOCK >>> (DEVICES[threadNum][0], Matrix2::DEVICES[threadNum][0], batchSize, length);
	B.copyToHost(0);
}

void MatrixBatch::print() {
	for (int n = 0; n < batchSize; n++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				printf("%f  ", host[n][e(i, j)]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

template<typename A>
void extendHostArray(A*& hostArray, int oldLength, int newLength) {
	A* newArray;
	cudaError_t err = cudaMallocHost(&newArray, newLength * sizeof(A));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	if (hostArray != NULL) {
		err = cudaMemcpy(newArray, hostArray, oldLength * sizeof(A), cudaMemcpyHostToHost);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
		err = cudaFreeHost(hostArray);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory deallocation failed: ") + cudaGetErrorString(err));
		}
	}
	hostArray = newArray;
}

template<typename A>
void extendDeviceArray(A*& deviceArray, int oldLength, int newLength) {
	A* newArray;
	cudaError_t err = cudaMalloc(&newArray, newLength * sizeof(A));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	if (deviceArray != NULL) {
		err = cudaMemcpy(newArray, deviceArray, oldLength * sizeof(A), cudaMemcpyDeviceToDevice);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
		err = cudaFree(deviceArray);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory deallocation failed: ") + cudaGetErrorString(err));
		}
	}
	deviceArray = newArray;
}

void MatrixBatch::copyToDevice(int deviceNum) {
	if (Utils::ALLOCATE_DEVICE_MODE) {
		if (deviceNum >= NUM_DEVICES) {
			extendHostArray(DEVICE_LENGTHS, NUM_DEVICES, deviceNum + 1);
			extendHostArray(DEVICE_BATCHSIZES, NUM_DEVICES, deviceNum + 1);
			for (int i = NUM_DEVICES; i <= deviceNum; i++) {
				DEVICE_LENGTHS[i] = 0;
				DEVICE_BATCHSIZES[i] = 0;
			}
			NUM_DEVICES = deviceNum + 1;
		} if (batchSize > DEVICE_BATCHSIZES[deviceNum]) {
			DEVICE_BATCHSIZES[deviceNum] = batchSize;
		} if (maxLength > DEVICE_LENGTHS[deviceNum]) {
			DEVICE_LENGTHS[deviceNum] = maxLength;
		}
		return;
	} else if (DEVICE_BATCHSIZES[deviceNum] < batchSize || DEVICE_LENGTHS[deviceNum] < length) {
		throw runtime_error("Device array of insufficient length");
	}
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(HOST_DEVICES[threadNum][deviceNum][i], host[i], length * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
}

void MatrixBatch::copyToDevice(int deviceNum, int threadNum) {
	if (Utils::ALLOCATE_DEVICE_MODE) {
		if (deviceNum >= NUM_DEVICES) {
			extendHostArray(DEVICE_LENGTHS, NUM_DEVICES, deviceNum + 1);
			extendHostArray(DEVICE_BATCHSIZES, NUM_DEVICES, deviceNum + 1);
			for (int i = NUM_DEVICES; i <= deviceNum; i++) {
				DEVICE_LENGTHS[i] = 0;
				DEVICE_BATCHSIZES[i] = 0;
			}
			NUM_DEVICES = deviceNum + 1;
		} if (batchSize > DEVICE_BATCHSIZES[deviceNum]) {
			DEVICE_BATCHSIZES[deviceNum] = batchSize;
		} if (maxLength > DEVICE_LENGTHS[deviceNum]) {
			DEVICE_LENGTHS[deviceNum] = maxLength;
		}
		return;
	}
	else if (DEVICE_BATCHSIZES[deviceNum] < batchSize || DEVICE_LENGTHS[deviceNum] < length) {
		throw runtime_error("Device array of insufficient length");
	}
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(HOST_DEVICES[threadNum][deviceNum][i], host[i], length * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
}

void MatrixBatch::copyToHost(int deviceNum) {
	if (Utils::ALLOCATE_DEVICE_MODE) {
		if (deviceNum >= NUM_DEVICES) {
			extendHostArray(DEVICE_LENGTHS, NUM_DEVICES, deviceNum + 1);
			extendHostArray(DEVICE_BATCHSIZES, NUM_DEVICES, deviceNum + 1);
			for (int i = NUM_DEVICES; i <= deviceNum; i++) {
				DEVICE_LENGTHS[i] = 0;
				DEVICE_BATCHSIZES[i] = 0;
			}
			NUM_DEVICES = deviceNum + 1;
		} if (batchSize > DEVICE_BATCHSIZES[deviceNum]) {
			DEVICE_BATCHSIZES[deviceNum] = batchSize;
		} if (maxLength > DEVICE_LENGTHS[deviceNum]) {
			DEVICE_LENGTHS[deviceNum] = maxLength;
		}
		return;
	} else if (deviceNum >= NUM_DEVICES || DEVICE_BATCHSIZES[deviceNum] < batchSize || DEVICE_LENGTHS[deviceNum] < length) {
		throw runtime_error("Device array of insufficient length");
	}
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(host[i], HOST_DEVICES[threadNum][deviceNum][i], length * sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
}

void MatrixBatch::copyToHost(int deviceNum, int copyLength) {
	if (Utils::ALLOCATE_DEVICE_MODE) {
		if (deviceNum >= NUM_DEVICES) {
			extendHostArray(DEVICE_LENGTHS, NUM_DEVICES, deviceNum + 1);
			extendHostArray(DEVICE_BATCHSIZES, NUM_DEVICES, deviceNum + 1);
			for (int i = NUM_DEVICES; i <= deviceNum; i++) {
				DEVICE_LENGTHS[i] = 0;
				DEVICE_BATCHSIZES[i] = 0;
			}
			NUM_DEVICES = deviceNum + 1;
		} if (batchSize > DEVICE_BATCHSIZES[deviceNum]) {
			DEVICE_BATCHSIZES[deviceNum] = batchSize;
		} if (maxLength > DEVICE_LENGTHS[deviceNum]) {
			DEVICE_LENGTHS[deviceNum] = maxLength;
		}
		return;
	} else if (deviceNum >= NUM_DEVICES || DEVICE_BATCHSIZES[deviceNum] < batchSize || DEVICE_LENGTHS[deviceNum] < length) {
		throw runtime_error("Device array of insufficient length");
	}
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(host[i], HOST_DEVICES[threadNum][deviceNum][i], copyLength * sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
}

void MatrixBatch::copyToHost(int deviceNum, int copyLength, int threadNum) {
	if (Utils::ALLOCATE_DEVICE_MODE) {
		if (deviceNum >= NUM_DEVICES) {
			extendHostArray(DEVICE_LENGTHS, NUM_DEVICES, deviceNum + 1);
			extendHostArray(DEVICE_BATCHSIZES, NUM_DEVICES, deviceNum + 1);
			for (int i = NUM_DEVICES; i <= deviceNum; i++) {
				DEVICE_LENGTHS[i] = 0;
				DEVICE_BATCHSIZES[i] = 0;
			}
			NUM_DEVICES = deviceNum + 1;
		} if (batchSize > DEVICE_BATCHSIZES[deviceNum]) {
			DEVICE_BATCHSIZES[deviceNum] = batchSize;
		} if (maxLength > DEVICE_LENGTHS[deviceNum]) {
			DEVICE_LENGTHS[deviceNum] = maxLength;
		}
		return;
	}
	else if (deviceNum >= NUM_DEVICES || DEVICE_BATCHSIZES[deviceNum] < batchSize || DEVICE_LENGTHS[deviceNum] < length) {
		throw runtime_error("Device array of insufficient length");
	}
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(host[i], HOST_DEVICES[threadNum][deviceNum][i], copyLength * sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
}

void MatrixBatch::copy(float* matrix) {
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(host[i], matrix, length * sizeof(float), cudaMemcpyHostToHost);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
}

void MatrixBatch::copy(MatrixBatch& B) {
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(host[i], B.host[i], length * sizeof(float), cudaMemcpyHostToHost);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
}

void MatrixBatch::copyTo(MatrixBatch& B) {
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(B.host[i], host[i], length * sizeof(float), cudaMemcpyDeviceToDevice);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
}

void MatrixBatch::copy(float** matrix) {
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < width; k++) {
				host[i][e(j, k)] = matrix[j][k];
			}
		}
	}
}

void MatrixBatch::setDims(int height, int width) {
	if (height * width > maxLength) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->height = height;
	this->width = width;
	this->length = height * width;
}

void MatrixBatch::setHeight(int height) {
	if (height * width > maxLength) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->height = height;
	this->length = height * width;
}

void MatrixBatch::setWidth(int width) {
	if (height * width > maxLength) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->width = width;
	this->length = height * width;
}

__global__
void kernelAdd(float** A, float** B, float** C, int N) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;

	if (n < N) {
		C[batch][n] = A[batch][n] + B[batch][n];
	}
}

void MatrixBatch::add(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C) {
	int N = A.length;
	int thread = max(A.threadNum, B.threadNum);
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1);
	dim3 blocks(numBlocks, A.batchSize);
	A.copyToDevice(0, thread);
	B.copyToDevice(1, thread);
	kernelAdd <<< blocks, Utils::THREADS_PER_BLOCK >>> (DEVICES[thread][0], DEVICES[thread][1], DEVICES[thread][2], N);
	C.copyToHost(2, N, thread);
}

__global__
void kernelMultiply(float** A, float** B, float** C, int N) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;

	if (n < N) {
		C[batch][n] = A[batch][n] * B[batch][n];
	}
}

void MatrixBatch::elementMultiply(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C) {
	int N = A.length;
	int thread = max(A.threadNum, B.threadNum);
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1);
	dim3 blocks(numBlocks, A.batchSize);
	A.copyToDevice(0, thread);
	B.copyToDevice(1, thread);
	kernelMultiply << < blocks, Utils::THREADS_PER_BLOCK >> > (DEVICES[thread][0], DEVICES[thread][1], DEVICES[thread][2], N);
	C.copyToHost(2, N, thread);
}

__global__
void kernelLinearCombo(float c1, float** A, float c2, float** B, float** C, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y;
	if (i < N) {
		C[b][i] = c1 * A[b][i] + c2 * B[b][i];
	}
}

void MatrixBatch::linearCombo(float c1, MatrixBatch& A, float c2, MatrixBatch& B, MatrixBatch& C) {
	int N = A.length;
	int thread = max(A.threadNum, B.threadNum);
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1);
	dim3 blocks(numBlocks, A.batchSize);
	A.copyToDevice(0, thread);
	B.copyToDevice(1, thread);
	kernelLinearCombo <<< blocks, Utils::THREADS_PER_BLOCK >> > (c1, DEVICES[thread][0], c2, DEVICES[thread][1], DEVICES[thread][2], N);
	C.copyToHost(2, N, thread);
}

void MatrixBatch::multiplyABC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) {
	int thread = max(A.threadNum, B.threadNum);
	A.copyToDevice(0, thread);
	B.copyToDevice(1, thread);
	C.copyToDevice(2, thread);
	cublasStatus_t stat = cublasSgemmBatched(Utils::HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, A.height, B.width, A.width, &ALPHA, DEVICES[thread][0], A.height, DEVICES[thread][1], B.height, &(overwrite ? BETA0 : BETA1), DEVICES[thread][2], C.height, C.batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost(2, A.height * B.width, thread);
}

void MatrixBatch::multiplyABC(Matrix2& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) {
	int thread = max(A.threadNum, B.threadNum);
	A.copyToBatchDevice(0, B.batchSize, thread);
	B.copyToDevice(1, thread);
	C.copyToDevice(2, thread);
	cublasStatus_t stat = cublasSgemmBatched(Utils::HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, A.height, B.width, A.width, &ALPHA, DEVICES[thread][0], A.height, DEVICES[thread][1], B.height, &(overwrite ? BETA0 : BETA1), DEVICES[thread][2], C.height, C.batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost(2, A.height * B.width, thread);
}

void MatrixBatch::multiplyAtBC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) {
	int thread = max(A.threadNum, B.threadNum);
	A.copyToDevice(0, thread);
	B.copyToDevice(1, thread);
	C.copyToDevice(2, thread);
	cublasStatus_t stat = cublasSgemmBatched(Utils::HANDLE, CUBLAS_OP_T, CUBLAS_OP_N, A.width, B.width, A.height, &ALPHA, DEVICES[thread][0], A.height, DEVICES[thread][1], B.height, &(overwrite ? BETA0 : BETA1), DEVICES[thread][2], C.height, C.batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost(2, A.width * B.width, thread);
}

void MatrixBatch::multiplyAtBC(Matrix2& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) {
	int thread = max(A.threadNum, B.threadNum);
	A.copyToBatchDevice(0, B.batchSize, thread);
	B.copyToDevice(1, thread);
	C.copyToDevice(2, thread);
	cublasStatus_t stat = cublasSgemmBatched(Utils::HANDLE, CUBLAS_OP_T, CUBLAS_OP_N, A.width, B.width, A.height, &ALPHA, DEVICES[thread][0], A.height, DEVICES[thread][1], B.height, &(overwrite ? BETA0 : BETA1), DEVICES[thread][2], C.height, C.batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost(2, A.width * B.width, thread);
}

void MatrixBatch::multiplyAtBtC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) {
	int thread = max(A.threadNum, B.threadNum);
	A.copyToDevice(0, thread);
	B.copyToDevice(1, thread);
	C.copyToDevice(2, thread);
	cublasStatus_t stat = cublasSgemmBatched(Utils::HANDLE, CUBLAS_OP_T, CUBLAS_OP_T, A.width, B.height, A.height, &ALPHA, DEVICES[thread][0], A.height, DEVICES[thread][1], B.height, &(overwrite ? BETA0 : BETA1), DEVICES[thread][2], C.height, C.batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost(2, A.width * B.height, thread);
}

void MatrixBatch::multiplyABtC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) {
	int thread = max(A.threadNum, B.threadNum);
	A.copyToDevice(0, thread);
	B.copyToDevice(1, thread);
	C.copyToDevice(2, thread);
	cublasStatus_t stat = cublasSgemmBatched(Utils::HANDLE, CUBLAS_OP_N, CUBLAS_OP_T, A.height, B.height, A.width, &ALPHA, DEVICES[thread][0], A.height, DEVICES[thread][1], B.height, &(overwrite ? BETA0 : BETA1), DEVICES[thread][2], C.height, C.batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost(2, A.height * B.height, thread);
}

MatrixBatch* MatrixBatch::allocateMatrixBatchArray(FillFunction& fill, int arrayLength, int batchSize, int height, int width) {
	MatrixBatch* array = new MatrixBatch[arrayLength];
	for (int i = 0; i < arrayLength; i++) {
		array[i] = MatrixBatch(fill, batchSize, height, width, i % Utils::NUM_THREADS);
	}
	return array;
}

MatrixBatch* MatrixBatch::allocateMatrixBatchArray(int arrayLength, int batchSize, int height, int width) {
	MatrixBatch* array = new MatrixBatch[arrayLength];
	for (int i = 0; i < arrayLength; i++) {
		array[i] = MatrixBatch(batchSize, height, width, i % Utils::NUM_THREADS);
	}
	return array;
}

long long MatrixBatch::allocateDevices() {
	long long numBytes = 0;
	cudaError_t err = cudaMallocHost(&HOST_DEVICES, Utils::NUM_THREADS * sizeof(float*));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	err = cudaMallocHost(&DEVICES, Utils::NUM_THREADS * sizeof(float*));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	for (int i = 0; i < Utils::NUM_THREADS; i++) {
		err = cudaMallocHost(&HOST_DEVICES[i], NUM_DEVICES * sizeof(float*));
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
		}
		err = cudaMallocHost(&DEVICES[i], NUM_DEVICES * sizeof(float*));
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
		}
		for (int j = 0; j < NUM_DEVICES; j++) {
			err = cudaMallocHost(&HOST_DEVICES[i][j], DEVICE_BATCHSIZES[j] * sizeof(float*));
			if (err != cudaSuccess) {
				throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
			}
			err = cudaMalloc(&DEVICES[i][j], DEVICE_BATCHSIZES[j] * sizeof(float*));
			if (err != cudaSuccess) {
				throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
			}
			numBytes += DEVICE_BATCHSIZES[j] * sizeof(float*);
			for (int k = 0; k < DEVICE_BATCHSIZES[j]; k++) {
				err = cudaMalloc(&HOST_DEVICES[i][j][k], DEVICE_LENGTHS[j] * sizeof(float));
				if (err != cudaSuccess) {
					throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
				}
				numBytes += DEVICE_LENGTHS[j] * sizeof(float);
			}
			err = cudaMemcpy(DEVICES[i][j], HOST_DEVICES[i][j], DEVICE_BATCHSIZES[j] * sizeof(float*), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
			}
		}
	}
	return numBytes;
}