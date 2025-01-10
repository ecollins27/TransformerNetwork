#include "Matrix.h"

Matrix::FillFunction* Matrix::ZERO_FILL{ new ConstantFill(0) };
Matrix::FillFunction* Matrix::UNIT_NORMAL_FILL{ new NormalFill(0,1) };
Matrix::FillFunction* Matrix::UNIT_UNIFORM_FILL{ new UniformFill(0,1) };

Matrix::Matrix(FillFunction* fillFunction, int height, int width, bool saveTranspose) {
	this->saveTranspose = saveTranspose;
	matrix = new float* [height];
	for (int i = 0; i < height; i++) {
		matrix[i] = new float[width];
		for (int j = 0; j < width; j++) {
			matrix[i][j] = fillFunction->operator()();
		}
	}
	if (saveTranspose) {
		matrixTrans = new float* [width];
		for (int i = 0; i < width; i++) {
			matrixTrans[i] = new float[height];
			for (int j = 0; j < height; j++) {
				matrixTrans[i][j] = matrix[j][i];
			}
		}
	}
	transposeUpdated = new bool(true);
}

Matrix::Matrix(float** matrix, float** matrixTrans) {
	this->matrix = matrix;
	this->matrixTrans = matrixTrans;
	saveTranspose = matrixTrans != NULL;
	if (saveTranspose) {
		transposeUpdated = new bool(true);
	}
}

float Matrix::operator()(int i, int j) {
	return matrix[i][j];
}

float& Matrix::r(int i, int j) {
	*transposeUpdated = false;
	return matrix[i][j];
}

void Matrix::calculateTranspose(int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			matrixTrans[j][i] = matrix[i][j];
		}
	}
	*transposeUpdated = true;
}

void Matrix::calculateMatrix(int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			matrix[i][j] = matrixTrans[j][i];
		}
	}
}

void Matrix::scale(int height, int width, float c) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			r(i,j) *= c;
		}
	}
}

void Matrix::copy(int height, int width, Matrix& to) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			to.r(i,j) = matrix[i][j];
		}
	}
}

void Matrix::fill(FillFunction* fillFunction, int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			r(i,j) = fillFunction->operator()();
		}
	}
}

void Matrix::print(int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%f ", matrix[i][j]);
		}
		printf("\n");
	}
}

Matrix Matrix::subMatrix(int i, int j, int height, int width) {
	Matrix sub;
	sub.saveTranspose = saveTranspose;
	sub.matrix = new float* [height];
	for (int a = 0; a < height; a++) {
		sub.matrix[a] = &matrix[i + a][j];
	}
	if (saveTranspose) {
		sub.matrixTrans = new float* [width];
		for (int a = 0; a < width; a++) {
			sub.matrixTrans[a] = &matrixTrans[j + a][i];
		}
		sub.transposeUpdated = transposeUpdated;
	}
	return sub;
}

float** Matrix::allocateMatrix(Matrix::FillFunction* fillFunction, int height, int width) {
	float** array = new float* [height];
	for (int i = 0; i < height; i++) {
		array[i] = new float[width];
		for (int j = 0; j < width; j++) {
			array[i][j] = fillFunction->operator()();
		}
	}
	return array;
}

bool Matrix::containsIllegalValue(int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (matrix[i][j] != matrix[i][j] || isinf(matrix[i][j])) {
				return true;
			}
		}
	}
	return false;
}

float*** Matrix::allocateMatrix3D(Matrix::FillFunction* fillFunction, int depth, int height, int width) {
	float*** array = new float** [depth];
	for (int i = 0; i < depth; i++) {
		array[i] = allocateMatrix(fillFunction, height, width);
	}
	return array;
}

Matrix* Matrix::allocateMatrixArray(Matrix::FillFunction* fillFunction, int x1, int x2, int x3, bool saveTranspose) {
	Matrix* array = new Matrix[x1];
	for (int i = 0; i < x1; i++) {
		array[i] = Matrix(fillFunction, x2, x3, saveTranspose);
	}
	return array;
}

Matrix** Matrix::allocateMatrixArray2D(Matrix::FillFunction* fillFunction, int x1, int x2, int x3, int x4, bool saveTranspose) {
	Matrix** array = new Matrix * [x1];
	for (int i = 0; i < x1; i++) {
		array[i] = new Matrix[x2];
		for (int j = 0; j < x2; j++) {
			array[i][j] = Matrix(fillFunction, x3, x4, saveTranspose);
		}
	}
	return array;
}

float Matrix::dotProduct(int n, float* a, float* b) {
	int i, n8 = n >> 3 << 3;
	__m128 vs1, vs2;
	float s, t[4];
	vs1 = _mm_setzero_ps();
	vs2 = _mm_setzero_ps();
	for (i = 0; i < n8; i += 8) {
		__m128 vx1, vx2, vy1, vy2;
		vx1 = _mm_loadu_ps(&a[i]);
		vx2 = _mm_loadu_ps(&a[i + 4]);
		vy1 = _mm_loadu_ps(&b[i]);
		vy2 = _mm_loadu_ps(&b[i + 4]);
		vs1 = _mm_add_ps(vs1, _mm_mul_ps(vx1, vy1));
		vs2 = _mm_add_ps(vs2, _mm_mul_ps(vx2, vy2));
	}
	for (s = 0.0f; i < n; ++i) s += a[i] * b[i];
	_mm_storeu_ps(t, vs1);
	s += t[0] + t[1] + t[2] + t[3];
	_mm_storeu_ps(t, vs2);
	s += t[0] + t[1] + t[2] + t[3];
	return s;
}

void Matrix::multiplyABC(int m, int n, int p, Matrix& A, Matrix& B, Matrix& C, bool overwrite) {
	if (!B.saveTranspose) {
		throw std::invalid_argument("B matrix must save transpose");
	} if (!*B.transposeUpdated) {
		B.calculateTranspose(n, p);
	}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C.r(i, j) = dotProduct(n, A.matrix[i], B.matrixTrans[j]);
			}
			else {
				C.r(i, j) += dotProduct(n, A.matrix[i], B.matrixTrans[j]);
			}
		}
	}
}

void Matrix::multiplyAtBC(int m, int n, int p, Matrix& A, Matrix& B, Matrix& C, bool overwrite) {
	if (!A.saveTranspose) {
		throw std::invalid_argument("A matrix must save transpose");
	} else if (!B.saveTranspose) {
		throw std::invalid_argument("B matrix must save transpose");
	}
	if (!*A.transposeUpdated) {
		A.calculateTranspose(n, m);
	} if (!*B.transposeUpdated) {
		B.calculateTranspose(n, p);
	}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C.r(i, j) = dotProduct(n, A.matrixTrans[i], B.matrixTrans[j]);
			}
			else {
				C.r(i, j) += dotProduct(n, A.matrixTrans[i], B.matrixTrans[j]);
			}
		}
	}
}

void Matrix::multiplyAtBtC(int m, int n, int p, Matrix& A, Matrix& B, Matrix& C, bool overwrite) {
	if (!A.saveTranspose) {
		throw std::invalid_argument("A matrix must save transpose");
	} if (!*A.transposeUpdated) {
		A.calculateTranspose(n, m);
	}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C.r(i, j) = dotProduct(n, A.matrixTrans[i], B.matrix[j]);
			}
			else {
				C.r(i, j) += dotProduct(n, A.matrixTrans[i], B.matrix[j]);
			}
		}
	}
}

void Matrix::multiplyABtC(int m, int n, int p, Matrix& A, Matrix& B, Matrix& C, bool overwrite) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C.r(i, j) = dotProduct(n, A.matrix[i], B.matrix[j]);
			}
			else {
				C.r(i, j) += dotProduct(n, A.matrix[i], B.matrix[j]);
			}
		}
	}
}

void Matrix::add(int m, int n, Matrix& A, Matrix& B, Matrix& C) {
	int n8 = n >> 3 << 3;
	__m128 a1, b1, a2, b2;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n8; j += 8) {
			a1 = _mm_loadu_ps(&A.matrix[i][j]);
			b1 = _mm_loadu_ps(&B.matrix[i][j]);
			a2 = _mm_loadu_ps(&A.matrix[i][j + 4]);
			b2 = _mm_loadu_ps(&B.matrix[i][j + 4]);
			_mm_store_ps(&C.matrix[i][j], _mm_add_ps(a1, b1));
			_mm_store_ps(&C.matrix[i][j + 4], _mm_add_ps(a2, b2));
		}
		for (int j = n8; j < n; j++) {
			C.matrix[i][j] = A.matrix[i][j] + B.matrix[i][j];
		}
	}
	if (C.saveTranspose) {
		*C.transposeUpdated = false;
	}
}

void Matrix::linearCombo(int m, int n, float c1, Matrix& A, float c2, Matrix& B, Matrix& C) {
	int n8 = n >> 3 << 3;
	__m128 cv1 = _mm_set1_ps(c1), cv2 = _mm_set1_ps(c2);
	__m128 a1, b1, a2, b2;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n8; j += 8) {
			a1 = _mm_loadu_ps(&A.matrix[i][j]);
			b1 = _mm_loadu_ps(&B.matrix[i][j]);
			a2 = _mm_loadu_ps(&A.matrix[i][j + 4]);
			b2 = _mm_loadu_ps(&B.matrix[i][j + 4]);
			_mm_store_ps(&C.matrix[i][j], _mm_add_ps(_mm_mul_ps(cv1, a1), _mm_mul_ps(cv2, b1)));
			_mm_store_ps(&C.matrix[i][j + 4], _mm_add_ps(_mm_mul_ps(cv1, a2), _mm_mul_ps(cv2, b2)));
		}
		for (int j = n8; j < n; j++) {
			C.matrix[i][j] = c1 * A.matrix[i][j] + c2 * B.matrix[i][j];
		}
	}
	if (C.saveTranspose) {
		*C.transposeUpdated = false;
	}
}

void Matrix::elementMultiply(int m, int n, Matrix& A, Matrix& B, Matrix& C) {
	int n8 = n >> 3 << 3;
	__m128 a1, b1, a2, b2;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n8; j += 8) {
			a1 = _mm_loadu_ps(&A.matrix[i][j]);
			b1 = _mm_loadu_ps(&B.matrix[i][j]);
			a2 = _mm_loadu_ps(&A.matrix[i][j + 4]);
			b2 = _mm_loadu_ps(&B.matrix[i][j + 4]);
			_mm_store_ps(&C.matrix[i][j], _mm_mul_ps(a1, b1));
			_mm_store_ps(&C.matrix[i][j + 4], _mm_mul_ps(a2, b2));
		}
		for (int j = n8; j < n; j++) {
			C.matrix[i][j] = A.matrix[i][j] * B.matrix[i][j];
		}
	}
	if (C.saveTranspose) {
		*C.transposeUpdated = false;
	}
}

Matrix::ConstantFill::ConstantFill(float value) {
	this->value = value;
}

float Matrix::ConstantFill::operator()() {
	return value;
}

Matrix::NormalFill::NormalFill(float mean, float stdDeviation) {
	distribution = { new normal_distribution<float>(mean, stdDeviation) };
}

float Matrix::NormalFill::operator()() {
	return (*distribution)(generator);
}

Matrix::UniformFill::UniformFill(float lowerBound, float upperBound) {
	distribution = { new uniform_real_distribution<float>(lowerBound, upperBound) };
}

float Matrix::UniformFill::operator()() {
	return (*distribution)(generator);
}