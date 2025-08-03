#include "Matrix2.h"

float Matrix2::ALPHA = 1.0f;
float Matrix2::BETA0 = 0.0f;
float Matrix2::BETA1 = 1.0f;
int Matrix2::NUM_DEVICES = 0;
int* Matrix2::DEVICE_LENGTHS = NULL;
float*** Matrix2::DEVICES = NULL;

Matrix2::Matrix2(int height, int width, int threadNum) {
	maxLength = height * width;
	this->length = maxLength;
	this->height = height;
	this->width = width;
	this->threadNum = threadNum;
	cudaError_t err = cudaMallocHost(&host, maxLength * sizeof(float));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	for (int i = 0; i < maxLength; i++) {
		host[i] = 0;
	}
}

Matrix2::Matrix2(FillFunction& fillFunction, int height, int width, int threadNum) {
	maxLength = height * width;
	this->length = maxLength;
	this->height = height;
	this->width = width;
	this->threadNum = threadNum;
	cudaError_t err = cudaMallocHost(&host, maxLength * sizeof(float));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	fill(fillFunction);
}

void Matrix2::free() {
	cudaError_t err = cudaFreeHost(host);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory deallocation failed: ") + cudaGetErrorString(err));
	}
}

int Matrix2::e(int i, int j) {
	return i + height * j;
}

float& Matrix2::operator()(int i, int j) {
	return host[e(i, j)];
}

void Matrix2::copy(float* host_matrix) {
	cudaError_t err = cudaMemcpy(host, host_matrix, length * sizeof(float), cudaMemcpyHostToHost);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
}

void Matrix2::copy(int height, int width, float** matrix) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			host[e(i, j)] = matrix[i][j];
		}
	}
}

void Matrix2::copy(Matrix2& B) {
	cudaError_t err = cudaMemcpy(host, B.host, length * sizeof(float), cudaMemcpyHostToHost);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
}

void Matrix2::copyTo(Matrix2& B) {
	cudaError_t err = cudaMemcpy(B.host, host, length * sizeof(float), cudaMemcpyHostToHost);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
}

__global__
void kernelTranspose(float* matrix, float* trans, int height1, int height2, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		int row = i % height1;
		int col = i / height1;
		trans[col + row * height2] = matrix[i];
	}
}
void Matrix2::transpose(Matrix2& B) {
	int N = length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	copyToDevice(0);
	kernelTranspose <<< numBlocks, Utils::THREADS_PER_BLOCK >>> (DEVICES[threadNum][0], DEVICES[threadNum][1], height, B.height, N);
	B.copyToHost(1, N);
}

void Matrix2::fill(FillFunction& fillFunction) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			host[e(i, j)] = fillFunction(i, j);
		}
	}
}

__global__
void kernelConstantFill(float c, float* matrix, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		matrix[i] = c;
	}
}

void Matrix2::constantFill(float c) {
	int N = length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelConstantFill <<< numBlocks, Utils::THREADS_PER_BLOCK >>> (c, DEVICES[threadNum][0], N);
	copyToHost(0);
}

__global__
void kernelScale(float c, float* matrix, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		matrix[i] *= c;
	}
}

void Matrix2::scale(float c) {
	int N = length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	copyToDevice(0);
	kernelScale <<< numBlocks, Utils::THREADS_PER_BLOCK >>> (c, DEVICES[threadNum][0], N);
	copyToHost(0);
}

__global__
void kernelSqrt(float* matrix, float* matrixSqrt, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		matrixSqrt[i] = sqrt(matrix[i]);
	}
}

void Matrix2::sqrt(Matrix2& B) {
	int N = length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	copyToDevice(0);
	kernelSqrt <<< numBlocks, Utils::THREADS_PER_BLOCK >>> (DEVICES[threadNum][0], DEVICES[threadNum][1], N);
	B.copyToHost(1, N);
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

void Matrix2::copyToBatchDevice(int deviceNum, int batchSize) {
	if (Utils::ALLOCATE_DEVICE_MODE) {
		if (deviceNum >= MatrixBatch::NUM_DEVICES) {
			extendHostArray(MatrixBatch::DEVICE_LENGTHS, MatrixBatch::NUM_DEVICES, deviceNum + 1);
			extendHostArray(MatrixBatch::DEVICE_BATCHSIZES, MatrixBatch::NUM_DEVICES, deviceNum + 1);
			for (int i = MatrixBatch::NUM_DEVICES; i <= deviceNum; i++) {
				MatrixBatch::DEVICE_LENGTHS[i] = 0;
				MatrixBatch::DEVICE_BATCHSIZES[i] = 0;
			}
			MatrixBatch::NUM_DEVICES = deviceNum + 1;
		} if (batchSize > MatrixBatch::DEVICE_BATCHSIZES[deviceNum]) {
			MatrixBatch::DEVICE_BATCHSIZES[deviceNum] = batchSize;
		} if (maxLength > MatrixBatch::DEVICE_LENGTHS[deviceNum]) {
			MatrixBatch::DEVICE_LENGTHS[deviceNum] = maxLength;
		}
		return;
	} else if (deviceNum >= MatrixBatch::NUM_DEVICES || MatrixBatch::DEVICE_BATCHSIZES[deviceNum] < batchSize || MatrixBatch::DEVICE_LENGTHS[deviceNum] < length) {
		throw runtime_error("Device array of insufficient length");
	}
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(MatrixBatch::HOST_DEVICES[threadNum][deviceNum][i], host, length * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
}

void Matrix2::copyToBatchDevice(int deviceNum, int batchSize, int threadNum) {
	if (Utils::ALLOCATE_DEVICE_MODE) {
		if (deviceNum >= MatrixBatch::NUM_DEVICES) {
			extendHostArray(MatrixBatch::DEVICE_LENGTHS, MatrixBatch::NUM_DEVICES, deviceNum + 1);
			extendHostArray(MatrixBatch::DEVICE_BATCHSIZES, MatrixBatch::NUM_DEVICES, deviceNum + 1);
			for (int i = MatrixBatch::NUM_DEVICES; i <= deviceNum; i++) {
				MatrixBatch::DEVICE_LENGTHS[i] = 0;
				MatrixBatch::DEVICE_BATCHSIZES[i] = 0;
			}
			MatrixBatch::NUM_DEVICES = deviceNum + 1;
		} if (batchSize > MatrixBatch::DEVICE_BATCHSIZES[deviceNum]) {
			MatrixBatch::DEVICE_BATCHSIZES[deviceNum] = batchSize;
		} if (maxLength > MatrixBatch::DEVICE_LENGTHS[deviceNum]) {
			MatrixBatch::DEVICE_LENGTHS[deviceNum] = maxLength;
		}
		return;
	}
	else if (deviceNum >= MatrixBatch::NUM_DEVICES || MatrixBatch::DEVICE_BATCHSIZES[deviceNum] < batchSize || MatrixBatch::DEVICE_LENGTHS[deviceNum] < length) {
		throw runtime_error("Device array of insufficient length");
	}
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(MatrixBatch::HOST_DEVICES[threadNum][deviceNum][i], host, length * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
}

void Matrix2::copyToDevice(int deviceNum) {
	if (Utils::ALLOCATE_DEVICE_MODE) {
		if (deviceNum >= NUM_DEVICES) {
			extendHostArray(DEVICE_LENGTHS, NUM_DEVICES, deviceNum + 1);
			for (int i = NUM_DEVICES; i <= deviceNum; i++) {
				DEVICE_LENGTHS[i] = 0;
			}
			NUM_DEVICES = deviceNum + 1;
		} if (maxLength > DEVICE_LENGTHS[deviceNum]) {
			DEVICE_LENGTHS[deviceNum] = maxLength;
		}
		return;
	} else if (deviceNum >= NUM_DEVICES || DEVICE_LENGTHS[deviceNum] < length) {
		throw runtime_error("Device array of insufficient length");
	}
	cudaError_t err = cudaMemcpy(DEVICES[threadNum][deviceNum], host, length * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
}

void Matrix2::copyToDevice(int deviceNum, int threadNum) {
	if (Utils::ALLOCATE_DEVICE_MODE) {
		if (deviceNum >= NUM_DEVICES) {
			extendHostArray(DEVICE_LENGTHS, NUM_DEVICES, deviceNum + 1);
			for (int i = NUM_DEVICES; i <= deviceNum; i++) {
				DEVICE_LENGTHS[i] = 0;
			}
			NUM_DEVICES = deviceNum + 1;
		} if (maxLength > DEVICE_LENGTHS[deviceNum]) {
			DEVICE_LENGTHS[deviceNum] = maxLength;
		}
		return;
	}
	else if (deviceNum >= NUM_DEVICES || DEVICE_LENGTHS[deviceNum] < length) {
		throw runtime_error("Device array of insufficient length");
	}
	cudaError_t err = cudaMemcpy(DEVICES[threadNum][deviceNum], host, length * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
}

void Matrix2::copyToHost(int deviceNum) {
	if (Utils::ALLOCATE_DEVICE_MODE) {
		if (deviceNum >= NUM_DEVICES) {
			extendHostArray(DEVICE_LENGTHS, NUM_DEVICES, deviceNum + 1);
			for (int i = NUM_DEVICES; i <= deviceNum; i++) {
				DEVICE_LENGTHS[i] = 0;
			}
			NUM_DEVICES = deviceNum + 1;
		} if (maxLength > DEVICE_LENGTHS[deviceNum]) {
			DEVICE_LENGTHS[deviceNum] = maxLength;
		}
		return;
	} else if (deviceNum >= NUM_DEVICES || DEVICE_LENGTHS[deviceNum] < length) {
		throw runtime_error("Device array of insufficient length");
	}
	cudaError_t err = cudaMemcpy(host, DEVICES[threadNum][deviceNum], length * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
}

void Matrix2::copyToHost(int deviceNum, int copyLength){
	if (Utils::ALLOCATE_DEVICE_MODE) {
		if (deviceNum >= NUM_DEVICES) {
			extendHostArray(DEVICE_LENGTHS, NUM_DEVICES, deviceNum + 1);
			for (int i = NUM_DEVICES; i <= deviceNum; i++) {
				DEVICE_LENGTHS[i] = 0;
			}
			NUM_DEVICES = deviceNum + 1;
		} if (maxLength > DEVICE_LENGTHS[deviceNum]) {
			DEVICE_LENGTHS[deviceNum] = maxLength;
		}
		return;
	} else if (deviceNum >= NUM_DEVICES || DEVICE_LENGTHS[deviceNum] < copyLength) {
		throw runtime_error("Device array of insufficient length");
	}
	else if (length < copyLength) {
		throw invalid_argument("copyLength must be less than or equal to host length");
	}
	cudaError_t err = cudaMemcpy(host, DEVICES[threadNum][deviceNum], copyLength * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
}

void Matrix2::copyToHost(int deviceNum, int copyLength, int threadNum) {
	if (Utils::ALLOCATE_DEVICE_MODE) {
		if (deviceNum >= NUM_DEVICES) {
			extendHostArray(DEVICE_LENGTHS, NUM_DEVICES, deviceNum + 1);
			for (int i = NUM_DEVICES; i <= deviceNum; i++) {
				DEVICE_LENGTHS[i] = 0;
			}
			NUM_DEVICES = deviceNum + 1;
		} if (maxLength > DEVICE_LENGTHS[deviceNum]) {
			DEVICE_LENGTHS[deviceNum] = maxLength;
		}
		return;
	}
	else if (deviceNum >= NUM_DEVICES || DEVICE_LENGTHS[deviceNum] < copyLength) {
		throw runtime_error("Device array of insufficient length");
	}
	else if (length < copyLength) {
		throw invalid_argument("copyLength must be less than or equal to host length");
	}
	cudaError_t err = cudaMemcpy(host, DEVICES[threadNum][deviceNum], copyLength * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
}

void Matrix2::print() {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%f  ", host[e(i, j)]);
		}
		printf("\n");
	}
}

void Matrix2::setDims(int height, int width) {
	if (height * width > maxLength) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->height = height;
	this->width = width;
	this->length = height * width;
}

void Matrix2::setHeight(int height) {
	if (height * width > maxLength) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->height = height;
	this->length = height * width;
}

void Matrix2::setWidth(int width) {
	if (height * width > maxLength) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->width = width;
	this->length = height * width;
}

MatrixBatch Matrix2::subMatrixBatch(int numMatrices, int subWidth) {
	MatrixBatch matrixBatch;
	matrixBatch.batchSize = numMatrices;
	matrixBatch.height = height;
	matrixBatch.width = subWidth;
	matrixBatch.maxLength = height * subWidth;
	matrixBatch.length = matrixBatch.maxLength;
	cudaError_t err = cudaMallocHost(&matrixBatch.host, numMatrices * sizeof(float*));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	for (int i = 0; i < numMatrices; i++) {
		matrixBatch.host[i] = &host[i * subWidth * height];
	}
	return matrixBatch;
}

void Matrix2::add(Matrix2& A, Matrix2& B, Matrix2& C) {
	int N = A.length;
	int N4 = N >> 2 << 2;
	__m128 a, b;
	for (int i = 0; i < N4; i += 4) {
		a = _mm_loadu_ps(&A.host[i]);
		b = _mm_loadu_ps(&B.host[i]);
		_mm_store_ps(&C.host[i], _mm_add_ps(a, b));
	}
	for (int i = N4; i < N; i++) {
		C.host[i] = A.host[i] + B.host[i];
	}
}

void Matrix2::add(int width, Matrix2& A, Matrix2& B, Matrix2& C) {
	int N = A.height * width;
	int N4 = N >> 2 << 2;
	__m128 a, b;
	for (int i = 0; i < N4; i += 4) {
		a = _mm_loadu_ps(&A.host[i]);
		b = _mm_loadu_ps(&B.host[i]);
		_mm_store_ps(&C.host[i], _mm_add_ps(a, b));
	}
	for (int i = N4; i < N; i++) {
		C.host[i] = A.host[i] + B.host[i];
	}
}

void Matrix2::elementMultiply(Matrix2& A, Matrix2& B, Matrix2& C) {
	int N = A.length;
	int N4 = N >> 2 << 2;
	__m128 a, b;
	for (int i = 0; i < N4; i += 4) {
		a = _mm_loadu_ps(&A.host[i]);
		b = _mm_loadu_ps(&B.host[i]);
		_mm_store_ps(&C.host[i], _mm_mul_ps(a, b));
	}
	for (int i = N4; i < N; i++) {
		C.host[i] = A.host[i] * B.host[i];
	}
}


void Matrix2::linearCombo(float c1, Matrix2& A, float c2, Matrix2& B, Matrix2& C) {
	return;
}

void Matrix2::multiplyABC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	int thread = max(A.threadNum, B.threadNum);
	A.copyToDevice(0, thread);
	B.copyToDevice(1, thread);
	if (!overwrite) {
		C.copyToDevice(2, thread);
	}
	if (!Utils::ALLOCATE_DEVICE_MODE) {
		cublasStatus_t stat = cublasSgemm(Utils::HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, A.height, B.width, A.width, &ALPHA, DEVICES[thread][0], A.height, DEVICES[thread][1], B.height, &(overwrite ? BETA0 : BETA1), DEVICES[thread][2], C.height);
		if (stat != CUBLAS_STATUS_SUCCESS) {
			throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
		}
	}
	C.copyToHost(2, A.height * B.width, thread);
}

void Matrix2::multiplyAtBC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	int thread = max(A.threadNum, B.threadNum);
	A.copyToDevice(0, thread);
	B.copyToDevice(1, thread);
	C.copyToDevice(2, thread);
	cublasStatus_t stat = cublasSgemm(Utils::HANDLE, CUBLAS_OP_T, CUBLAS_OP_N, A.width, B.width, A.height, &ALPHA, DEVICES[thread][0], A.height, DEVICES[thread][1], B.height, &(overwrite ? BETA0 : BETA1), DEVICES[thread][2], C.height);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost(2, A.width * B.width, thread);
}

void Matrix2::multiplyAtBtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	int thread = max(A.threadNum, B.threadNum);
	A.copyToDevice(0, thread);
	B.copyToDevice(1, thread);
	C.copyToDevice(2, thread);
	cublasStatus_t stat = cublasSgemm(Utils::HANDLE, CUBLAS_OP_T, CUBLAS_OP_T, A.width, B.height, A.height, &ALPHA, DEVICES[thread][0], A.height, DEVICES[thread][1], B.height, &(overwrite ? BETA0 : BETA1), DEVICES[thread][2], C.height);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost(2, A.width * B.height, thread);
}

void Matrix2::multiplyABtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	int thread = max(A.threadNum, B.threadNum);
	A.copyToDevice(0, thread);
	B.copyToDevice(1, thread);
	C.copyToDevice(2, thread);
	cublasStatus_t stat = cublasSgemm(Utils::HANDLE, CUBLAS_OP_N, CUBLAS_OP_T, A.height, B.height, A.width, &ALPHA, DEVICES[thread][0], A.height, DEVICES[thread][1], B.height, &(overwrite ? BETA0 : BETA1), DEVICES[thread][2], C.height);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost(2, A.height * B.height, thread);
}

Matrix2* Matrix2::allocateMatrixArray(FillFunction& fillFunction, int batchSize, int height, int width) {
	Matrix2* array = new Matrix2[batchSize];
	for (int i = 0; i < batchSize; i++) {
		array[i] = Matrix2(fillFunction, height, width, i % Utils::NUM_THREADS);
	}
	return array;
}

Matrix2* Matrix2::allocateMatrixArray(int batchSize, int height, int width) {
	Matrix2* array = new Matrix2[batchSize];
	for (int i = 0; i < batchSize; i++) {
		array[i] = Matrix2(height, width, i % Utils::NUM_THREADS);
	}
	return array;
}

long long Matrix2::allocateDevices() {
	long long numBytes = 0;
	cudaError_t err = cudaMallocHost(&DEVICES, Utils::NUM_THREADS * sizeof(float*));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	for (int i = 0; i < Utils::NUM_THREADS; i++) {
		err = cudaMallocHost(&DEVICES[i], NUM_DEVICES * sizeof(float*));
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
		}
		for (int j = 0; j < NUM_DEVICES; j++) {
			err = cudaMalloc(&DEVICES[i][j], DEVICE_LENGTHS[j] * sizeof(float));
			if (err != cudaSuccess) {
				throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
			}
			numBytes += DEVICE_LENGTHS[j] * sizeof(float);
		}
	}
	return numBytes;
}