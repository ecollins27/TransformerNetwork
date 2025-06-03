#include "Matrix2.h"

float Matrix2::ALPHA = 1.0f;
float Matrix2::BETA = 0.0f;
int Matrix2::THREADS_PER_BLOCK = 256;
cublasHandle_t Matrix2::HANDLE = NULL;

Matrix2::Matrix2(int height, int width) {
	maxHeight = height;
	maxWidth = width;
	this->height = height;
	this->width = width;
	cudaError_t err = cudaMalloc(&device, maxHeight * maxWidth * sizeof(float));
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
	allocateHost();
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
	cudaMemcpy(device, host_matrix, height * width * sizeof(float), cudaMemcpyHostToDevice);
	copyToHost();
}

void Matrix2::allocateHost() {
	cudaMallocHost(&host, maxHeight * maxWidth * sizeof(float));
	cudaMemcpy(host, device, height * width * sizeof(float), cudaMemcpyDeviceToHost);
}

void Matrix2::deallocateHost() {
	copyToDevice();
	cudaFreeHost(&host);
	host = NULL;
}

void Matrix2::copyToDevice() {
	copy(host);
}

void Matrix2::copyToHost() {
	cudaMemcpy(host, device, height * width * sizeof(float), cudaMemcpyDeviceToHost);
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
	if (height > maxHeight || width > maxWidth) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->height = height;
	this->width = width;
}

float Matrix2::rowColumnDot(Matrix2& A, Matrix2& B, int i, int j) {
	int n4 = A.width >> 2 << 2;
	float s = 0.0f, t[4];
	__m128 vs = _mm_setzero_ps();
	__m128 vx, vy;
	for (int k = 0; k < n4; k += 4) {
		vx = _mm_loadu_ps(&A.host[A.width * i + k]);
		vy = _mm_setr_ps(B.host[B.e(k, j)], B.host[B.e(k + 1, j)], B.host[B.e(k + 2, j)], B.host[B.e(k + 3, j)]);
		vs = _mm_add_ps(vs, _mm_mul_ps(vx, vy));
	}
	for (int k = n4; k < A.width; k++) {
		s += A.host[A.width * i + k] * B.host[B.e(k, j)];
	}
	_mm_storeu_ps(t, vs);
	s += t[0] + t[1] + t[2] + t[3];
	return s;
}

void Matrix2::simdMultiplyABC(Matrix2& A, Matrix2& B, Matrix2& C) {
	for (int i = 0; i < A.height; i++) {
		for (int j = 0; j < B.width; j++) {
			C.host[C.e(i, j)] = rowColumnDot(A, B, i, j);
		}
	}
	C.copyToDevice();
}

void Matrix2::add(Matrix2& A, Matrix2& B, Matrix2& C) {
	int N = A.height * A.width;
	int N4 = N >> 2 << 2;
	__m128 va, vb;
	for (int i = 0; i < N4; i += 4) {
		va = _mm_loadu_ps(&A.host[i]);
		vb = _mm_loadu_ps(&B.host[i]);
		_mm_storeu_ps(&C.host[i], _mm_add_ps(va, vb));
	}
	for (int i = N4; i < N; i++) {
		C.host[i] = A.host[i] + B.host[i];
	}
	C.copyToDevice();
}

void Matrix2::elementMultiply(Matrix2& A, Matrix2& B, Matrix2& C) {
	int N = A.height * A.width;
	int N4 = N >> 2 << 2;
	__m128 va, vb;
	for (int i = 0; i < N4; i += 4) {
		va = _mm_loadu_ps(&A.host[i]);
		vb = _mm_loadu_ps(&B.host[i]);
		_mm_storeu_ps(&C.host[i], _mm_mul_ps(va, vb));
	}
	for (int i = N4; i < N; i++) {
		C.host[i] = A.host[i] * B.host[i];
	}
	C.copyToDevice();
}

void Matrix2::multiplyABC(Matrix2& A, Matrix2& B, Matrix2& C) {
	cublasStatus_t stat = cublasSgemm(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, C.width, C.height, A.width, &ALPHA, B.device, C.width, A.device, A.width, &BETA, C.device, C.width);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("cuBLAS multiplication failed");
	}
	C.copyToHost();
}