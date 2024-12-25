#include "Matrix2.h"

Matrix2::FillFunction* Matrix2::ZERO_FILL{ new ConstantFill(0) };
Matrix2::FillFunction* Matrix2::UNIT_NORMAL_FILL{ new NormalFill(0,1) };
Matrix2::FillFunction* Matrix2::UNIT_UNIFORM_FILL{ new UniformFill(0,1) };

Matrix2::Matrix2(FillFunction* fillFunction, int height, int width, bool saveTranspose) {
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
}

Matrix2::Matrix2(float** matrix, float** matrixTrans) {
	this->matrix = matrix;
	this->matrixTrans = matrixTrans;
	saveTranspose = matrixTrans != NULL;
}

float& Matrix2::operator()(int i, int j) {
	transposeUpdated = false;
	return matrix[i][j];
}

void Matrix2::calculateTranspose(int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			matrixTrans[j][i] = matrix[i][j];
		}
	}
	transposeUpdated = true;
}

void Matrix2::calculateMatrix(int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			matrix[i][j] = matrixTrans[j][i];
		}
	}
}

void Matrix2::scale(int height, int width, float c) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			matrix[i][j] *= c;
		}
	}
}

void Matrix2::copy(int height, int width, Matrix2& to) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			to.matrix[i][j] = matrix[i][j];
		}
	}
}

void Matrix2::fill(FillFunction* fillFunction, int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			matrix[i][j] = fillFunction->operator()();
		}
	}
}

void Matrix2::print(int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%f ", matrix[i][j]);
		}
		printf("\n");
	}
}

Matrix2 Matrix2::subMatrix(int i, int j, int height, int width) {
	Matrix2 sub;
	sub.saveTranspose = saveTranspose;
	sub.matrix = new float* [height];
	for (int a = 0; a < height; a++) {
		sub.matrix[a] = &matrix[i + a][j];
	}
	if (saveTranspose) {
		sub.matrixTrans = new float* [width];
		for (int a = 0; a < width; a++) {
			sub.matrixTrans[a] = &matrix[j + a][i];
		}
	}
	return sub;
}

Matrix2* Matrix2::allocateMatrixArray(Matrix2::FillFunction* fillFunction, int x1, int x2, int x3, bool saveTranspose) {
	Matrix2* array = new Matrix2[x1];
	for (int i = 0; i < x1; i++) {
		array[i] = Matrix2(fillFunction, x2, x3, saveTranspose);
	}
	return array;
}

Matrix2** Matrix2::allocateMatrixArray2D(Matrix2::FillFunction* fillFunction, int x1, int x2, int x3, int x4, bool saveTranspose) {
	Matrix2** array = new Matrix2 * [x1];
	for (int i = 0; i < x1; i++) {
		array[i] = new Matrix2[x2];
		for (int j = 0; j < x2; j++) {
			array[i][j] = Matrix2(fillFunction, x3, x4, saveTranspose);
		}
	}
	return array;
}

float Matrix2::dotProduct(int n, float* a, float* b) {
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
	for (s = 0.0f; i < n; ++i) s += a[i] * a[i];
	_mm_storeu_ps(t, vs1);
	s += t[0] + t[1] + t[2] + t[3];
	_mm_storeu_ps(t, vs2);
	s += t[0] + t[1] + t[2] + t[3];
	return s;
}

void Matrix2::multiplyABC(int m, int n, int p, Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	if (!B.saveTranspose) {
		throw std::invalid_argument("B matrix must save transpose");
	} if (!B.transposeUpdated) {
		B.calculateTranspose(n, p);
	}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C(i, j) = dotProduct(n, A.matrix[i], B.matrixTrans[j]);
			}
			else {
				C(i, j) += dotProduct(n, A.matrix[i], B.matrixTrans[j]);
			}
		}
	}
}

void Matrix2::multiplyAtBC(int m, int n, int p, Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	if (!A.saveTranspose) {
		throw std::invalid_argument("A matrix must save transpose");
	} else if (!B.saveTranspose) {
		throw std::invalid_argument("B matrix must save transpose");
	}
	if (!A.transposeUpdated) {
		A.calculateTranspose(n, m);
	} if (!B.transposeUpdated) {
		B.calculateTranspose(n, p);
	}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C(i, j) = dotProduct(n, A.matrixTrans[i], B.matrixTrans[j]);
			}
			else {
				C(i, j) += dotProduct(n, A.matrixTrans[i], B.matrixTrans[j]);
			}
		}
	}
}

void Matrix2::multiplyAtBtC(int m, int n, int p, Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	if (!A.saveTranspose) {
		throw std::invalid_argument("A matrix must save transpose");
	} if (!A.transposeUpdated) {
		A.calculateTranspose(n, m);
	}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C(i, j) = dotProduct(n, A.matrixTrans[i], B.matrix[j]);
			}
			else {
				C(i, j) += dotProduct(n, A.matrixTrans[i], B.matrix[j]);
			}
		}
	}
}

void Matrix2::multiplyABtC(int m, int n, int p, Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C(i, j) = dotProduct(n, A.matrix[i], B.matrix[j]);
			}
			else {
				C(i, j) += dotProduct(n, A.matrix[i], B.matrix[j]);
			}
		}
	}
}

void Matrix2::elementAdd(int m, int n, Matrix2& A, Matrix2& B, Matrix2& C, float c1, float c2, bool overwrite) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (overwrite) {
				C(i, j) = A(i, j) + B(i, j);
			}
			else {
				C(i, j) += A(i, j) + B(i, j);
			}
		}
	}
}
void Matrix2::elementMultiply(int m, int n, Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (overwrite) {
				C(i, j) = A(i, j) * B(i, j);
			}
			else {
				C(i, j) += A(i, j) * B(i, j);
			}
		}
	}
}

Matrix2::ConstantFill::ConstantFill(float value) {
	this->value = value;
}

float Matrix2::ConstantFill::operator()() {
	return value;
}

Matrix2::NormalFill::NormalFill(float mean, float stdDeviation) {
	distribution = { new normal_distribution<float>(mean, stdDeviation) };
}

float Matrix2::NormalFill::operator()() {
	return (*distribution)(generator);
}

Matrix2::UniformFill::UniformFill(float lowerBound, float upperBound) {
	distribution = { new uniform_real_distribution<float>(lowerBound, upperBound) };
}

float Matrix2::UniformFill::operator()() {
	return (*distribution)(generator);
}