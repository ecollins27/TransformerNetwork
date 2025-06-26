#include "Matrix2.h"

float Matrix2::ALPHA = 1.0f;
float Matrix2::BETA0 = 0.0f;
float Matrix2::BETA1 = 1.0f;
cublasHandle_t Matrix2::HANDLE = NULL;

Matrix2::Matrix2(int height, int width, bool allocateHost) {
	maxLength = height * width;
	this->height = height;
	this->width = width;
	cudaError_t err = cudaMalloc(&device, maxLength * sizeof(float));
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
	if (allocateHost) {
		this->allocateHost();
	}
	else {
		host = NULL;
	}
}

Matrix2::Matrix2(FillFunction& fillFunction, int height, int width) {
	maxLength = height * width;
	this->height = height;
	this->width = width;
	cudaError_t err = cudaMalloc(&device, maxLength * sizeof(float));
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
	allocateHost();
	fill(fillFunction);
}

int Matrix2::e(int i, int j) {
	return i * width + j;
}

float& Matrix2::operator()(int i, int j) {
	if (host == NULL) {
		throw invalid_argument("Matrix must allocate host");
	}
	return host[e(i, j)];
}

void Matrix2::copy(float* host_matrix) {
	cudaError_t err = cudaMemcpy(device, host_matrix, height * width * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory copy failed");
	}
	copyToHost();
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
	int N = height * width;
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
	int N = height * width;
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
	int N = height * width;
	int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	kernelSqrt <<< numBlocks, THREADS_PER_BLOCK >>> (device, N);
	copyToHost();
}

__global__
void kernelMean(float* input, float* output, int M, int N) {
	extern __shared__ float shared[];

	int row = blockIdx.x;
	int tid = threadIdx.x;
	int stride = blockDim.x;

	float sum = 0.0f;
	for (int i = tid; i < N; i += stride) {
		sum += input[row * N + i];
	}

	shared[tid] = sum;
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shared[tid] += shared[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		output[row] = shared[0] / N;
	}
}

void Matrix2::mean(Matrix2& mean) {
	kernelMean <<< height, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float*) >>> (device, mean.device, height, width);
	mean.copyToHost();
}

__global__
void kernelVariance(float* matrix, float* mean, float* output, int M, int N) {
	extern __shared__ float shared[];

	int row = blockIdx.x;
	int tid = threadIdx.x;
	int stride = blockDim.x;

	float sum = 0.0f;
	for (int i = tid; i < N; i += stride) {
		sum += (matrix[row * N + i] - mean[row]) * (matrix[row * N + i] - mean[row]);
	}

	shared[tid] = sum;
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shared[tid] += shared[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		output[row] = shared[0] / N;
	}
}

void Matrix2::variance(Matrix2& mean, Matrix2& variance) {
	kernelVariance <<< height, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float*) >>> (device, mean.device, variance.device, height, width);
	variance.copyToHost();
}

__global__
void kernelNormalize(float* matrix, float* mean, float* std, float* output, int N, int width) {

	int num = blockIdx.x * blockDim.x + threadIdx.x;
	int row = num / width;

	if (num >= N) {
		return;
	}
	if (std[num] == 0) {
		output[num] = (matrix[num] - mean[row]) / (0.0000001);
	}
	else {
		output[num] = (matrix[num] - mean[row]) / std[row];
	}
}

void Matrix2::normalize(Matrix2& mean, Matrix2& std, Matrix2& normalizedOutput) {
	int N = height * width;
	int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	kernelNormalize <<< numBlocks, THREADS_PER_BLOCK >>> (device, mean.device, std.device, normalizedOutput.device, N, width);
	normalizedOutput.copyToHost();
}

void Matrix2::allocateHost() {
	cudaError_t err = cudaMallocHost(&host, maxLength * sizeof(float));
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
	err = cudaMemcpy(host, device, height * width * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory copy failed");
	}
}

void Matrix2::deallocateHost() {
	copyToDevice();
	cudaError_t err = cudaFreeHost(&host);
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory deallocation failed");
	}
	host = NULL;
}

void Matrix2::copyToDevice() {
	cudaError_t err = cudaMemcpy(device, host, height * width * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory copy failed");
	}
}

void Matrix2::copyToHost() {
	cudaError_t err = cudaMemcpy(host, device, height * width * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory copy failed");
	}
}

void Matrix2::print() {
	if (host == NULL) {
		throw invalid_argument("Matrix must allocate host before printing");
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
}

void Matrix2::setHeight(int height) {
	if (height * width > maxLength) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->height = height;
}

void Matrix2::setWidth(int width) {
	if (height * width > maxLength) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->width = width;
}

MatrixBatch Matrix2::subMatrixBatch(int numMatrices, int subHeight) {
	MatrixBatch matrixBatch;
	matrixBatch.batchSize = numMatrices;
	matrixBatch.height = subHeight;
	matrixBatch.width = width;
	cudaError_t err = cudaMalloc(&matrixBatch.device, numMatrices * sizeof(float*));
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
	err = cudaMallocHost(&matrixBatch.host, numMatrices * sizeof(float*));
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
	err = cudaMallocHost(&matrixBatch.hostDevice, numMatrices * sizeof(float*));
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
	for (int i = 0; i < numMatrices; i++) {
		matrixBatch.host[i] = &host[i * subHeight * width];
		matrixBatch.hostDevice[i] = device + (i * subHeight * width * sizeof(float));
	}
	err = cudaMemcpy(matrixBatch.device, matrixBatch.hostDevice, numMatrices * sizeof(float*), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory copy failed");
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
	int N = A.height * A.width;
	int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	kernelAdd <<< numBlocks, THREADS_PER_BLOCK >>> (N, A.device, B.device, C.device);
	C.copyToHost();
}

void Matrix2::elementMultiply(Matrix2& A, Matrix2& B, Matrix2& C) {
	int N = A.height * A.width;
	int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	kernelMultiply <<< numBlocks, THREADS_PER_BLOCK >>> (N, A.device, B.device, C.device);
	C.copyToHost();
}

void Matrix2::multiplyABC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	cublasStatus_t stat = cublasSgemm(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, B.width, A.height, A.width, &ALPHA, B.device, B.width, A.device, A.width, &(overwrite? BETA0:BETA1), C.device, C.width);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("cuBLAS multiplication failed");
	}
	C.copyToHost();
}

void Matrix2::multiplyAtBC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	cublasStatus_t stat = cublasSgemm(HANDLE, CUBLAS_OP_N, CUBLAS_OP_T, B.width, A.width, A.height, &ALPHA, B.device, B.width, A.device, A.width, &(overwrite? BETA0 : BETA1), C.device, C.width);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("cuBLAS multiplication failed");
	}
	C.copyToHost();
}

void Matrix2::multiplyAtBtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	cublasStatus_t stat = cublasSgemm(HANDLE, CUBLAS_OP_T, CUBLAS_OP_T, B.height, A.width, A.height, &ALPHA, B.device, B.width, A.device, A.width, &(overwrite ? BETA0 : BETA1), C.device, C.width);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("cuBLAS multiplication failed");
	}
	C.copyToHost();
}

void Matrix2::multiplyABtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	cublasStatus_t stat = cublasSgemm(HANDLE, CUBLAS_OP_T, CUBLAS_OP_N, B.height, A.height, A.width, &ALPHA, B.device, B.width, A.device, A.width, &(overwrite ? BETA0 : BETA1), C.device, C.width);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("cuBLAS multiplication failed");
	}
	C.copyToHost();
}