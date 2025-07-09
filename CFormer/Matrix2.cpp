#include "Matrix2.h"

float Matrix2::ALPHA = 1.0f;
float Matrix2::BETA0 = 0.0f;
float Matrix2::BETA1 = 1.0f;

// TODO: Refactor to store matrix column wise instead of row wise
Matrix2::Matrix2(int height, int width, bool allocateHost) {
	maxLength = height * width;
	this->length = maxLength;
	this->height = height;
	this->width = width;
	cudaError_t err = cudaMalloc(&device, maxLength * sizeof(float));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	if (allocateHost) {
		this->allocateHost();
	}
	else {
		host = NULL;
	}
	constantFill(0);
}

Matrix2::Matrix2(FillFunction& fillFunction, int height, int width) {
	maxLength = height * width;
	this->length = maxLength;
	this->height = height;
	this->width = width;
	cudaError_t err = cudaMalloc(&device, maxLength * sizeof(float));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	allocateHost();
	fill(fillFunction);
}

void Matrix2::free() {
	cudaError_t err;
	if (batchDevice != NULL) {
		err = cudaFree(batchDevice);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory deallocation failed: ") + cudaGetErrorString(err));
		}
	} if (host != NULL) {
		err = cudaFreeHost(host);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory deallocation failed: ") + cudaGetErrorString(err));
		}
	}
	err = cudaFree(device);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory deallocation failed: ") + cudaGetErrorString(err));
	}
}

int Matrix2::e(int i, int j) {
	return i + height * j;
}

float& Matrix2::operator()(int i, int j) {
	if (host == NULL) {
		throw invalid_argument("Matrix must allocate host");
	}
	return host[e(i, j)];
}

void Matrix2::copy(float* host_matrix) {
	cudaError_t err = cudaMemcpy(device, host_matrix, length * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
	copyToHost();
}

void Matrix2::copy(float** matrix) {
	if (host == NULL) {
		throw invalid_argument("Matrix must allocate host");
	}
	cudaError_t err;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			host[e(i, j)] = matrix[i][j];
		}
	}
	copyToDevice();
}

void Matrix2::copy(Matrix2& B) {
	cudaError_t err = cudaMemcpy(device, B.device, length * sizeof(float), cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
	copyToHost();
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
void Matrix2::transpose(Matrix2& B, int N) {
	int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	kernelTranspose <<< numBlocks, THREADS_PER_BLOCK >>> (device, B.device, height, B.height, N);
	B.copyToHost();
}

void Matrix2::fill(FillFunction& fillFunction) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			host[e(i, j)] = fillFunction(i, j);
		}
	}
	copyToDevice();
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
	int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	kernelConstantFill <<< numBlocks, THREADS_PER_BLOCK >>> (c, device, N);
	copyToHost();
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
	int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	kernelScale <<< numBlocks, THREADS_PER_BLOCK >>> (c, device, N);
	copyToHost();
}

__global__
void kernelSqrt(float* matrix, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		matrix[i] = sqrt(matrix[i]);
	}
}

void Matrix2::sqrt(Matrix2& B) {
	int N = length;
	int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	kernelSqrt <<< numBlocks, THREADS_PER_BLOCK >>> (device, N);
	copyToHost();
}

void Matrix2::allocateHost() {
	cudaError_t err = cudaMallocHost(&host, maxLength * sizeof(float));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	err = cudaMemcpy(host, device, length * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
}

void Matrix2::deallocateHost() {
	copyToDevice();
	cudaError_t err = cudaFreeHost(&host);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory deallocation failed: ") + cudaGetErrorString(err));
	}
	host = NULL;
}

void Matrix2::allocateBatchDevice(int batchSize) {
	float** hostDevice;
	cudaMallocHost(&hostDevice, batchSize * sizeof(float*));
	cudaMalloc(&batchDevice, batchSize * sizeof(float*));
	for (int i = 0; i < batchSize; i++) {
		hostDevice[i] = device;
	}
	cudaMemcpy(batchDevice, hostDevice, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
	cudaFreeHost(hostDevice);
}

void Matrix2::deallocateBatchDevice(int batchSize) {
	cudaFree(batchDevice);
}

void Matrix2::copyToDevice() {
	cudaError_t err = cudaMemcpy(device, host, length * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
}

void Matrix2::copyToHost() {
	cudaError_t err = cudaMemcpy(host, device, length * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
}

void Matrix2::print() {
	if (host == NULL) {
		throw invalid_argument("Matrix must allocate host");
	}
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
	cudaError_t err = cudaMalloc(&matrixBatch.device, numMatrices * sizeof(float*));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	err = cudaMallocHost(&matrixBatch.hostDevice, numMatrices * sizeof(float*));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	for (int i = 0; i < numMatrices; i++) {
		matrixBatch.host[i] = &host[i * subWidth * height];
		matrixBatch.hostDevice[i] = device + (i * subWidth * height * sizeof(float));
	}
	err = cudaMemcpy(matrixBatch.device, matrixBatch.hostDevice, numMatrices * sizeof(float*), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
	return matrixBatch;
}


__global__
void kernelAdd(int N, float* A, float* B, float* C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = A[i] + B[i];
	}
}

__global__
void kernelMultiply(int N, float* A, float* B, float* C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = A[i] * B[i];
	}
}

void Matrix2::add(Matrix2& A, Matrix2& B, Matrix2& C) {
	int N = A.length;
	int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	kernelAdd <<< numBlocks, THREADS_PER_BLOCK >>> (N, A.device, B.device, C.device);
	C.copyToHost();
}

void Matrix2::add(int width, Matrix2& A, Matrix2& B, Matrix2& C) {
	int N = A.height * width;
	int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	kernelAdd << < numBlocks, THREADS_PER_BLOCK >> > (N, A.device, B.device, C.device);
	C.copyToHost();
}

void Matrix2::elementMultiply(Matrix2& A, Matrix2& B, Matrix2& C) {
	int N = A.length;
	int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	kernelMultiply <<< numBlocks, THREADS_PER_BLOCK >>> (N, A.device, B.device, C.device);
	C.copyToHost();
}

__global__
void kernelLinearCombo(float c1, float* A, float c2, float* B, float* C, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = c1 * A[i] + c2 * B[i];
	}
}


void Matrix2::linearCombo(float c1, Matrix2& A, float c2, Matrix2& B, Matrix2& C) {
	int N = A.length;
	int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	kernelLinearCombo << < numBlocks, THREADS_PER_BLOCK >> > (c1, A.device, c2, B.device, C.device, N);
	C.copyToHost();
}

void Matrix2::multiplyABC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	//cublasStatus_t stat = cublasSgemm(MatrixBatch::HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, B.width, A.height, A.width, &ALPHA, B.device, B.width, A.device, A.width, &(overwrite? BETA0:BETA1), C.device, C.width);
	cublasStatus_t stat = cublasSgemm(MatrixBatch::HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, A.height, B.width, A.width, &ALPHA, A.device, A.height, B.device, B.height, &(overwrite ? BETA0 : BETA1), C.device, C.height);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost();
}

void Matrix2::multiplyAtBC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	//cublasStatus_t stat = cublasSgemm(MatrixBatch::HANDLE, CUBLAS_OP_N, CUBLAS_OP_T, B.width, A.width, A.height, &ALPHA, B.device, B.width, A.device, A.width, &(overwrite? BETA0 : BETA1), C.device, C.width);
	cublasStatus_t stat = cublasSgemm(MatrixBatch::HANDLE, CUBLAS_OP_T, CUBLAS_OP_N, A.width, B.width, A.height, &ALPHA, A.device, A.height, B.device, B.height, &(overwrite ? BETA0 : BETA1), C.device, C.height);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost();
}

void Matrix2::multiplyAtBtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	//cublasStatus_t stat = cublasSgemm(MatrixBatch::HANDLE, CUBLAS_OP_T, CUBLAS_OP_T, B.height, A.width, A.height, &ALPHA, B.device, B.width, A.device, A.width, &(overwrite ? BETA0 : BETA1), C.device, C.width);
	cublasStatus_t stat = cublasSgemm(MatrixBatch::HANDLE, CUBLAS_OP_T, CUBLAS_OP_T, A.width, B.height, A.height, &ALPHA, A.device, A.height, B.device, B.height, &(overwrite ? BETA0 : BETA1), C.device, C.height);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost();
}

void Matrix2::multiplyABtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	//cublasStatus_t stat = cublasSgemm(MatrixBatch::HANDLE, CUBLAS_OP_T, CUBLAS_OP_N, B.height, A.height, A.width, &ALPHA, B.device, B.width, A.device, A.width, &(overwrite ? BETA0 : BETA1), C.device, C.width);
	cublasStatus_t stat = cublasSgemm(MatrixBatch::HANDLE, CUBLAS_OP_N, CUBLAS_OP_T, A.height, B.height, A.width, &ALPHA, A.device, A.height, B.device, B.height, &(overwrite ? BETA0 : BETA1), C.device, C.height);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost();
}

Matrix2* Matrix2::allocateMatrixArray(FillFunction& fillFunction, int batchSize, int height, int width) {
	Matrix2* array = new Matrix2[batchSize];
	for (int i = 0; i < batchSize; i++) {
		array[i] = Matrix2(fillFunction, height, width);
	}
	return array;
}

Matrix2* Matrix2::allocateMatrixArray(int batchSize, int height, int width, bool allocateHost) {
	Matrix2* array = new Matrix2[batchSize];
	for (int i = 0; i < batchSize; i++) {
		array[i] = Matrix2(height, width, allocateHost);
	}
	return array;
}