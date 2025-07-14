#include "MatrixBatch.h"
#include "Matrix2.h"

float MatrixBatch::ALPHA = 1.0f;
float MatrixBatch::BETA0 = 0.0f;
float MatrixBatch::BETA1 = 1.0f;
cublasHandle_t MatrixBatch::HANDLE = NULL;

MatrixBatch::MatrixBatch(int batchSize, int height, int width, bool allocateHost) {
	maxLength = length;
	this->length = maxLength;
	this->batchSize = batchSize;
	this->height = height;
	this->width = width;
	cudaError_t err = cudaMalloc(&device, batchSize * sizeof(float*));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
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
	this->length = maxLength;
	this->batchSize = batchSize;
	this->height = height;
	this->width = width;
	cudaError_t err = cudaMalloc(&device, batchSize * sizeof(float*));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	err = cudaMallocHost(&hostDevice, batchSize * sizeof(float*));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	for (int i = 0; i < batchSize; i++) {
		err = cudaMalloc(&hostDevice[i], maxLength * sizeof(float));
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
		}
	}
	err = cudaMemcpy(device, hostDevice, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
	allocateHost();
	fill(fillFunction);
}

void MatrixBatch::free() {
	cudaError_t err;
	if (host != NULL) {
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
	for (int i = 0; i < batchSize; i++) {
		err = cudaFree(hostDevice[i]);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory deallocation failed: ") + cudaGetErrorString(err));
		}
	}
	err = cudaFreeHost(hostDevice);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory deallocation failed: ") + cudaGetErrorString(err));
	}
	err = cudaFree(device);
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
	int N = length;
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
	int N = length;
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
	int N = length;
	int numBlocks = (N + THREADS_PER_BLOCK - 1);
	dim3 blocks(numBlocks, batchSize);
	kernelSqrt <<< blocks, THREADS_PER_BLOCK >> > (device, B.device, N);
	copyToHost();
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
	int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	kernelCondense <<< numBlocks, THREADS_PER_BLOCK >>> (device, B.device, batchSize, length);
	B.copyToHost();
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
			throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
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
			throw runtime_error(string("CUDA memory deallocation failed: ") + cudaGetErrorString(err));
		}
	}
	err = cudaFreeHost(&host);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory deallocation failed: ") + cudaGetErrorString(err));
	}
	host = NULL;
}

void MatrixBatch::copyToDevice() {
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(hostDevice[i], host[i], length * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
}

void MatrixBatch::copyToHost() {
	if (host == NULL) {
		return;
	}
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(host[i], hostDevice[i], length * sizeof(float), cudaMemcpyDeviceToHost);
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
		err = cudaMemcpy(hostDevice[i], matrix, length * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
}

void MatrixBatch::copy(MatrixBatch& B) {
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(hostDevice[i], B.hostDevice[i], length * sizeof(float), cudaMemcpyDeviceToDevice);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
	copyToHost();
}

void MatrixBatch::copyTo(MatrixBatch& B) {
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(B.hostDevice[i], hostDevice[i], length * sizeof(float), cudaMemcpyDeviceToDevice);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
	B.copyToHost();
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
	copyToDevice();
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
	int N = A.length;
	int numBlocks = (N + THREADS_PER_BLOCK - 1);
	dim3 blocks(numBlocks, A.batchSize);
	kernelMultiply << < blocks, THREADS_PER_BLOCK >> > (A.device, B.device, C.device, N);
	C.copyToHost();
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
	int numBlocks = (N + THREADS_PER_BLOCK - 1);
	dim3 blocks(numBlocks, A.batchSize);
	kernelLinearCombo <<< blocks, THREADS_PER_BLOCK >> > (c1, A.device, c2, B.device, C.device, N);
	C.copyToHost();
}

void MatrixBatch::multiplyABC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) {
	//cublasStatus_t stat = cublasSgemmBatched(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, B.width, A.height, A.width, &ALPHA, B.device, B.width, A.device, A.width, &(overwrite ? BETA0 : BETA1), C.device, C.width, C.batchSize);
	cublasStatus_t stat = cublasSgemmBatched(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, A.height, B.width, A.width, &ALPHA, A.device, A.height, B.device, B.height, &(overwrite ? BETA0 : BETA1), C.device, C.height, C.batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost();
}

void MatrixBatch::multiplyABC(Matrix2& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) {
	//cublasStatus_t stat = cublasSgemmBatched(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, B.width, A.height, A.width, &ALPHA, B.device, B.width, A.batchDevice, A.width, &(overwrite ? BETA0 : BETA1), C.device, C.width, C.batchSize);
	cublasStatus_t stat = cublasSgemmBatched(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, A.height, B.width, A.width, &ALPHA, A.batchDevice, A.height, B.device, B.height, &(overwrite ? BETA0 : BETA1), C.device, C.height, C.batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost();
}

void MatrixBatch::multiplyAtBC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) {
	//cublasStatus_t stat = cublasSgemmBatched(HANDLE, CUBLAS_OP_N, CUBLAS_OP_T, B.width, A.width, A.height, &ALPHA, B.device, B.width, A.device, A.width, &(overwrite ? BETA0 : BETA1), C.device, C.width, C.batchSize);
	cublasStatus_t stat = cublasSgemmBatched(HANDLE, CUBLAS_OP_T, CUBLAS_OP_N, A.width, B.width, A.height, &ALPHA, A.device, A.height, B.device, B.height, &(overwrite ? BETA0 : BETA1), C.device, C.height, C.batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost();
}

void MatrixBatch::multiplyAtBC(Matrix2& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) {
	//cublasStatus_t stat = cublasSgemmBatched(HANDLE, CUBLAS_OP_N, CUBLAS_OP_T, B.width, A.width, A.height, &ALPHA, B.device, B.width, A.batchDevice, A.width, &(overwrite ? BETA0 : BETA1), C.device, C.width, C.batchSize);
	cublasStatus_t stat = cublasSgemmBatched(HANDLE, CUBLAS_OP_T, CUBLAS_OP_N, A.width, B.width, A.height, &ALPHA, A.batchDevice, A.height, B.device, B.height, &(overwrite ? BETA0 : BETA1), C.device, C.height, C.batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost();
}

void MatrixBatch::multiplyAtBtC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) {
	//cublasStatus_t stat = cublasSgemmBatched(HANDLE, CUBLAS_OP_T, CUBLAS_OP_T, B.height, A.width, A.height, &ALPHA, B.device, B.width, A.device, A.width, &(overwrite ? BETA0 : BETA1), C.device, C.width, C.batchSize);
	cublasStatus_t stat = cublasSgemmBatched(HANDLE, CUBLAS_OP_T, CUBLAS_OP_T, A.width, B.height, A.height, &ALPHA, A.device, A.height, B.device, B.height, &(overwrite ? BETA0 : BETA1), C.device, C.height, C.batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost();
}

void MatrixBatch::multiplyABtC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) {
	//cublasStatus_t stat = cublasSgemmBatched(HANDLE, CUBLAS_OP_T, CUBLAS_OP_N, B.height, A.height, A.width, &ALPHA, B.device, B.width, A.device, A.width, &(overwrite ? BETA0 : BETA1), C.device, C.width, C.batchSize);
	cublasStatus_t stat = cublasSgemmBatched(HANDLE, CUBLAS_OP_N, CUBLAS_OP_T, A.height, B.height, A.width, &ALPHA, A.device, A.height, B.device, B.height, &(overwrite ? BETA0 : BETA1), C.device, C.height, C.batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	C.copyToHost();
}

MatrixBatch* MatrixBatch::allocateMatrixBatchArray(FillFunction& fill, int arrayLength, int batchSize, int height, int width) {
	MatrixBatch* array = new MatrixBatch[arrayLength];
	for (int i = 0; i < arrayLength; i++) {
		array[i] = MatrixBatch(fill, batchSize, height, width);
	}
	return array;
}

MatrixBatch* MatrixBatch::allocateMatrixBatchArray(int arrayLength, int batchSize, int height, int width, bool allocateHost) {
	MatrixBatch* array = new MatrixBatch[arrayLength];
	for (int i = 0; i < arrayLength; i++) {
		array[i] = MatrixBatch(batchSize, height, width, allocateHost);
	}
	return array;
}