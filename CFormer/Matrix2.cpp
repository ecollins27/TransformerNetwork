#include "Matrix2.h"

float Matrix2::ALPHA = 1.0f;
float Matrix2::BETA0 = 0.0f;
float Matrix2::BETA1 = 1.0f;
int Matrix2::THREADS_PER_BLOCK = 256;
cublasHandle_t Matrix2::HANDLE = NULL;
Matrix2::ConstantFill Matrix2::ZERO_FILL = ConstantFill(0);
Matrix2::NormalFill Matrix2::UNIT_NORMAL_FILL = NormalFill(0,1);
Matrix2::UniformFill Matrix2::UNIT_UNIFORM_FILL = UniformFill(0,1);

Matrix2::Matrix2(int height, int width) {
	maxLength = height * width;
	this->height = height;
	this->width = width;
	cudaError_t err = cudaMalloc(&device, maxLength * sizeof(float));
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
	allocateHost();
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
		throw invalid_argument("Matrix must be converted to host before accessing");
	}
	return host[e(i, j)];
}

void Matrix2::copy(float* host_matrix) {
	cudaError_t err = cudaMemcpy(device, host_matrix, height * width * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
	copyToHost();
}

void Matrix2::fill(FillFunction& fillFunction) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			host[width * i + j] = fillFunction(i, j);
		}
	}
	copyToDevice();
}

void Matrix2::constantFill(float c) {
	int n = height * width;
	int n4 = n >> 2 << 2;
	__m128 C = _mm_set1_ps(c);
	for (int i = 0; i < n4; i += 4) {
		_mm_store_ps(&host[i], C);
	}
	for (int i = n4; i < n; i++) {
		host[i] = c;
	}
	copyToDevice();
}

void Matrix2::scale(float c) {
	int n = height * width;
	int n4 = n >> 2 << 2;
	__m128 C = _mm_set1_ps(c);
	for (int i = 0; i < n4; i += 4) {
		_mm_store_ps(&host[i], _mm_mul_ps(_mm_loadu_ps(&host[i]), C));
	}
	for (int i = n4; i < n; i++) {
		host[i] *= c;
	}
	copyToDevice();
}

void Matrix2::sqrt(Matrix2& B, int num) {
	int n = height * width;
	int n4 = n >> 2 << 2;
	for (int i = 0; i < n4; i += 4) {
		_mm_store_ps(&B.host[i], _mm_sqrt_ps(_mm_loadu_ps(&host[i])));
	}
	for (int i = n4; i < n; i++) {
		B.host[i] = std::sqrt(host[i]);
	}
	B.copyToDevice();
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
void kernelStd(float* matrix, float* mean, float* output, int M, int N) {
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

void Matrix2::std(Matrix2& mean, Matrix2& std) {
	kernelStd <<< height, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float*) >> > (device, mean.device, std.device, height, width);
	std.copyToHost();
}

void Matrix2::allocateHost() {
	cudaError_t err = cudaMallocHost(&host, maxLength * sizeof(float));
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
	err = cudaMemcpy(host, device, height * width * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
}

void Matrix2::deallocateHost() {
	copyToDevice();
	cudaError_t err = cudaFreeHost(&host);
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
	host = NULL;
}

void Matrix2::copyToDevice() {
	cudaError_t err = cudaMemcpy(device, host, height * width * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
}

void Matrix2::copyToHost() {
	cudaError_t err = cudaMemcpy(host, device, height * width * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
}

void Matrix2::print() {
	if (host == NULL) {
		throw invalid_argument("Matrix must be converted to host before printing");
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
	err = cudaMallocHost(&matrixBatch.deviceArray, numMatrices * sizeof(float*));
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
	for (int i = 0; i < numMatrices; i++) {
		matrixBatch.host[i] = &host[i * subHeight * width];
		matrixBatch.deviceArray[i] = device + (i * subHeight * width * sizeof(float));
	}
	err = cudaMemcpy(matrixBatch.device, matrixBatch.deviceArray, numMatrices * sizeof(float*), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
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
	cublasStatus_t stat = cublasSgemm(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, B.width, A.height, A.width, &ALPHA, B.device, B.width, A.device, A.width, &(overwrite? BETA0:BETA1), C.device, B.width);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("cuBLAS multiplication failed");
	}
	C.copyToHost();
}

void Matrix2::multiplyAtBC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	cublasStatus_t stat = cublasSgemm(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, B.width, A.height, A.height, &ALPHA, B.device, B.width, A.device, A.height, &(overwrite ? BETA0 : BETA1), C.device, B.width);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("cuBLAS multiplication failed");
	}
	C.copyToHost();
}

void Matrix2::multiplyAtBtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	cublasStatus_t stat = cublasSgemm(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, B.width, A.height, A.height, &ALPHA, B.device, B.height, A.device, A.height, &(overwrite ? BETA0 : BETA1), C.device, B.width);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("cuBLAS multiplication failed");
	}
	C.copyToHost();
}

void Matrix2::multiplyABtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	cublasStatus_t stat = cublasSgemm(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, B.width, A.height, A.width, &ALPHA, B.device, B.height, A.device, A.width, &(overwrite ? BETA0 : BETA1), C.device, B.width);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("cuBLAS multiplication failed");
	}
	C.copyToHost();
}

Matrix2::ConstantFill::ConstantFill(float value) {
	this->value = value;
}

float Matrix2::ConstantFill::operator()(int i, int j) {
	return value;
}

Matrix2::NormalFill::NormalFill(float mean, float stdDeviation) {
	distribution = new normal_distribution<float>(mean, stdDeviation);
}

float Matrix2::NormalFill::operator()(int i, int j) {
	return (*distribution)(generator);
}

Matrix2::UniformFill::UniformFill(float lowerBound, float upperBound) {
	distribution = new uniform_real_distribution<float>(lowerBound, upperBound);
}

float Matrix2::UniformFill::operator()(int i, int j) {
	return (*distribution)(generator);
}