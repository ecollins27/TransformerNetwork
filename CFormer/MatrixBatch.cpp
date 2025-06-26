#include "MatrixBatch.h"

float MatrixBatch::ALPHA = 1.0f;
float MatrixBatch::BETA0 = 0.0f;
float MatrixBatch::BETA1 = 1.0f;
cublasHandle_t MatrixBatch::HANDLE = NULL;

MatrixBatch::MatrixBatch(int batchSize, int height, int width, bool allocateHost) {
	maxLength = height * width;
	this->batchSize = batchSize;
	this->height = height;
	this->width = width;
	cudaError_t err = cudaMalloc(&device, batchSize * sizeof(float*));
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
	err = cudaMallocHost(&hostDevice, batchSize * sizeof(float*));
	for (int i = 0; i < batchSize; i++) {
		cudaMalloc(&hostDevice[i], maxLength * sizeof(float));
	}
	cudaMemcpy(device, hostDevice, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
	if (allocateHost) {
		this->allocateHost();
	}
	else {
		host = NULL;
	}
}

MatrixBatch::MatrixBatch(FillFunction& fillFunction, int batchSize, int height, int width) {
	maxLength = height * width;
	this->batchSize = batchSize;
	this->height = height;
	this->width = width;
	cudaError_t err = cudaMalloc(&device, batchSize * sizeof(float*));
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
	err = cudaMallocHost(&hostDevice, batchSize * sizeof(float*));
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
	for (int i = 0; i < batchSize; i++) {
		err = cudaMalloc(&hostDevice[i], maxLength * sizeof(float));
		if (err != cudaSuccess) {
			throw invalid_argument("CUDA memory allocation failed");
		}
	}
	err = cudaMemcpy(device, hostDevice, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory copy failed");
	}
	allocateHost();
	fill(fillFunction);
}

int MatrixBatch::e(int i, int j) {
	return width * i + j;
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
	copyToDevice();
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
	int N = height * width;
	int numBlocks = (N + THREADS_PER_BLOCK - 1);
	dim3 blocks(numBlocks, batchSize);
	kernelConstantFill <<< blocks, THREADS_PER_BLOCK >>> (c, device, N);
	copyToHost();
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
	int N = height * width;
	int numBlocks = (N + THREADS_PER_BLOCK - 1);
	dim3 blocks(numBlocks, batchSize);
	kernelScale << < blocks, THREADS_PER_BLOCK >> > (c, device, N);
	copyToHost();
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
	int N = height * width;
	int numBlocks = (N + THREADS_PER_BLOCK - 1);
	dim3 blocks(numBlocks, batchSize);
	kernelSqrt <<< blocks, THREADS_PER_BLOCK >> > (device, B.device, N);
	copyToHost();
}

void MatrixBatch::print() {
	if (host == NULL) {
		throw invalid_argument("Matrix must allocate host before printing");
	}
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

void MatrixBatch::allocateHost() {
	cudaError_t err = cudaMallocHost(&host, batchSize * sizeof(float*));
	for (int i = 0; i < batchSize; i++) {
		err = cudaMallocHost(&host[i], maxLength * sizeof(float));
		if (err != cudaSuccess) {
			throw invalid_argument("CUDA memory allocation failed");
		}
	}
	copyToHost();
}

void MatrixBatch::deallocateHost() {
	copyToDevice();
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		cudaFreeHost(&host[i]);
		if (err != cudaSuccess) {
			throw invalid_argument("CUDA memory deallocation failed");
		}
	}
	err = cudaFreeHost(&host);
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory deallocation failed");
	}
	host = NULL;
}

void MatrixBatch::copyToDevice() {
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(hostDevice[i], host[i], height * width * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			throw invalid_argument("CUDA memory copy failed");
		}
	}
}

void MatrixBatch::copyToHost() {
	if (host == NULL) {
		return;
	}
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(host[i], hostDevice[i], height * width * sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			throw invalid_argument("CUDA memory copy failed");
		}
	}
}

void MatrixBatch::copy(float* matrix) {
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(host[i], matrix, height * width * sizeof(float), cudaMemcpyHostToHost);
		if (err != cudaSuccess) {
			throw invalid_argument("CUDA memory copy failed");
		}
		err = cudaMemcpy(hostDevice[i], matrix, height * width * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			throw invalid_argument("CUDA memory copy failed");
		}
	}
}

void MatrixBatch::copy(float** matrix) {
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(host[i], matrix[i], height * width * sizeof(float), cudaMemcpyHostToHost);
		if (err != cudaSuccess) {
			throw invalid_argument("CUDA memory copy failed");
		}
		err = cudaMemcpy(hostDevice[i], matrix[i], height * width * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			throw invalid_argument("CUDA memory copy failed");
		}
	}
}

void MatrixBatch::setDims(int height, int width) {
	if (height * width > maxLength) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->height = height;
	this->width = width;
}

void MatrixBatch::setHeight(int height) {
	if (height * width > maxLength) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->height = height;
}

void MatrixBatch::setWidth(int width) {
	if (height * width > maxLength) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->width = width;
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
	int N = A.height * A.width;
	int numBlocks = (N + THREADS_PER_BLOCK - 1);
	dim3 blocks(numBlocks, A.batchSize);
	kernelAdd <<< blocks, THREADS_PER_BLOCK >>> (A.device, B.device, C.device, N);
	C.copyToHost();
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
	int N = A.height * A.width;
	int numBlocks = (N + THREADS_PER_BLOCK - 1);
	dim3 blocks(numBlocks, A.batchSize);
	kernelMultiply << < blocks, THREADS_PER_BLOCK >> > (A.device, B.device, C.device, N);
	C.copyToHost();
}

void MatrixBatch::multiplyABC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) {
	cublasStatus_t stat = cublasSgemmBatched(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, B.width, A.height, A.width, &ALPHA, B.device, B.width, A.device, A.width, &(overwrite ? BETA0 : BETA1), C.device, C.width, C.batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("cuBLAS multiplication failed");
	}
	C.copyToHost();
}

void MatrixBatch::multiplyAtBC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) {
	cublasStatus_t stat = cublasSgemmBatched(HANDLE, CUBLAS_OP_N, CUBLAS_OP_T, B.width, A.width, A.height, &ALPHA, B.device, B.width, A.device, A.width, &(overwrite ? BETA0 : BETA1), C.device, C.width, C.batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("cuBLAS multiplication failed");
	}
	C.copyToHost();
}

void MatrixBatch::multiplyAtBtC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) {
	cublasStatus_t stat = cublasSgemmBatched(HANDLE, CUBLAS_OP_T, CUBLAS_OP_T, B.height, A.width, A.height, &ALPHA, B.device, B.width, A.device, A.width, &(overwrite ? BETA0 : BETA1), C.device, C.width, C.batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("cuBLAS multiplication failed");
	}
	C.copyToHost();
}

void MatrixBatch::multiplyABtC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) {
	cublasStatus_t stat = cublasSgemmBatched(HANDLE, CUBLAS_OP_T, CUBLAS_OP_N, B.height, A.height, A.width, &ALPHA, B.device, B.width, A.device, A.width, &(overwrite ? BETA0 : BETA1), C.device, C.width, C.batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("cuBLAS multiplication failed");
	}
	C.copyToHost();
}