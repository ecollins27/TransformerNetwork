#include "Matrix.h"

Matrix::FillFunction* Matrix::ZERO_FILL{ new ConstantFill(0) };
Matrix::FillFunction* Matrix::UNIT_NORMAL_FILL{ new NormalFill(0,1) };
Matrix::FillFunction* Matrix::UNIT_UNIFORM_FILL{ new UniformFill(0,1) };

float Matrix::dotProduct(int n, float* x, float* y) {
	int i, n8 = n >> 3 << 3;
	__m128 vs1, vs2;
	float s, t[4];
	vs1 = _mm_setzero_ps();
	vs2 = _mm_setzero_ps();
	for (i = 0; i < n8; i += 8) {
		__m128 vx1, vx2, vy1, vy2;
		vx1 = _mm_loadu_ps(&x[i]);
		vx2 = _mm_loadu_ps(&x[i + 4]);
		vy1 = _mm_loadu_ps(&y[i]);
		vy2 = _mm_loadu_ps(&y[i + 4]);
		vs1 = _mm_add_ps(vs1, _mm_mul_ps(vx1, vy1));
		vs2 = _mm_add_ps(vs2, _mm_mul_ps(vx2, vy2));
	}
	for (s = 0.0f; i < n; ++i) s += x[i] * y[i];
	_mm_storeu_ps(t, vs1);
	s += t[0] + t[1] + t[2] + t[3];
	_mm_storeu_ps(t, vs2);
	s += t[0] + t[1] + t[2] + t[3];
	return s;
}

float** Matrix::allocateMatrix(FillFunction* fillFunction, int height, int width) {
	float** matrix = (float**)malloc(height * sizeof(float*));
	for (int i = 0; i < height; i++) {
		matrix[i] = (float*)malloc(width * sizeof(float));
		for (int j = 0; j < width; j++) {
			matrix[i][j] = fillFunction->get();
		}
	}
	return matrix;
}

float*** Matrix::allocate3DMatrix(FillFunction* fillFunction, int d1, int d2, int d3) {
	float*** array = (float***)malloc(d1 * sizeof(float**));
	for (int i = 0; i < d1; i++) {
		array[i] = allocateMatrix(fillFunction, d2, d3);
	}
	return array;
}

float**** Matrix::allocate4DMatrix(FillFunction* fillFunction, int d1, int d2, int d3, int d4){
	float**** array = (float****)malloc(d1 * sizeof(float***));
	for (int i = 0; i < d1; i++) {
		array[i] = allocate3DMatrix(fillFunction, d2, d3, d4);
	}
	return array;
}

void Matrix::deallocateMatrix(float** A, int height, int width) {
	for (int i = 0; i < height; i++) {
		free(A[i]);
	}
	free(A);
}

void Matrix::deallocate3DMatrix(float*** A, int d1, int d2, int d3) {
	for (int i = 0; i < d1; i++) {
		deallocateMatrix(A[i], d2, d3);
	}
	free(A);
}

void Matrix::deallocate4DMatrix(float**** A, int d1, int d2, int d3, int d4) {
	for (int i = 0; i < d1; i++) {
		deallocate3DMatrix(A[i], d2, d3, d4);
	}
	free(A);
}

bool Matrix::containsNaN(int height, int width, float** A) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (A[i][j] != A[i][j]) {
				return true;
			}
			else if (isinf(A[i][j])) {
				return true;
			}
		}
	}
	return false;
}

void Matrix::add(int m, int n, float** A, float** B, float** C, float scalar1, float scalar2) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			C[i][j] = scalar1 * A[i][j] + scalar2 * B[i][j];
		}
	}
}

void Matrix::scale(int m, int n, float** A, float scalar) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			A[i][j] *= scalar;
		}
	}
}

void Matrix::transpose(int m, int n, float** A, float** At) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			At[j][i] = A[i][j];
		}
	}
}

void Matrix::transposeInPlace(int m, float** A) {
	float temp;
	for (int i = 0; i < m; i++) {
		for (int j = i + 1; j < m; j++) {
			temp = A[i][j];
			A[i][j] = A[j][i];
			A[j][i] = temp;
		}
	}
}

void Matrix::matrixMultiplyABC(int m, int n, int p, float** A, float** B, float** C, bool overwrite) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C[i][j] = 0;
			}
			for (int k = 0; k < n; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

void Matrix::matrixMultiplyAtBC(int m, int n, int p, float** A, float** B, float** C, bool overwrite) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C[i][j] = 0;
			}
			for (int k = 0; k < n; k++) {
				C[i][j] += A[k][i] * B[k][j];
			}
		}
	}
}

void Matrix::subMatrixMultiplyABtC(int m, int n, int p, float** A, float** B, float** C, bool overwrite, int startY) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C[i][j] = dotProduct(n, &A[i][startY], B[j]);
			}
			else {
				C[i][j] += dotProduct(n, &A[i][startY], B[j]);
			}
		}
	}
}

void Matrix::matrixMultiplyABtC(int m, int n, int p, float** A, float** B, float** C, bool overwrite){
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C[i][j] = dotProduct(n, A[i], B[j]);
			}
			else {
				C[i][j] += dotProduct(n, A[i], B[j]);
			}
		}
	}
}

void Matrix::matrixMultiplyABtCt(int m, int n, int p, float** A, float** B, float** C, bool overwrite) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C[j][i] = dotProduct(n, A[i], B[j]);
			}
			else {
				C[j][i] += dotProduct(n, A[i], B[j]);
			}
		}
	}
}

void Matrix::matrixTensorMultiply(int m, int n, int p, float** A, float*** B, float** C, bool overwrite) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C[i][j] = dotProduct(n, A[i], B[i][j]);
			} else {
				C[i][j] += dotProduct(n, A[i], B[i][j]);
			}
		}
	}
}

void Matrix::elementMultiply(int m, int n, float** A, float** B, float** C, bool overwrite) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (overwrite) {
				C[i][j] = A[i][j] * B[i][j];
			}
			else {
				C[i][j] += A[i][j] * B[i][j];
			}
		}
	}
}

void Matrix::fill(FillFunction* fillFunction, int m, int n, float** A) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			A[i][j] = fillFunction->get();
		}
	}
}

void Matrix::copy(int m, int n, float** from, float** to) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			to[i][j] = from[i][j];
		}
	}
}

void Matrix::print(int m, int n, float** A) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			printf("%f  ", A[i][j]);
		}
		printf("\n");
	}
}

Matrix::ConstantFill::ConstantFill(float value){
	this->value = value;
}

float Matrix::ConstantFill::get() {
	return value;
}

Matrix::NormalFill::NormalFill(float mean, float stdDeviation) {
	distribution = { new normal_distribution<float>(mean, stdDeviation) };
}

float Matrix::NormalFill::get() {
	return (*distribution)(generator);
}

Matrix::UniformFill::UniformFill(float lowerBound, float upperBound){
	distribution = { new uniform_real_distribution<float>(lowerBound, upperBound) };
}

float Matrix::UniformFill::get() {
	return (*distribution)(generator);
}