#include "Matrix.h"

Matrix::FillFunction* Matrix::ZERO_FILL{ new ConstantFill(0) };
Matrix::FillFunction* Matrix::UNIT_NORMAL_FILL{ new NormalFill(0,1) };
Matrix::FillFunction* Matrix::UNIT_UNIFORM_FILL{ new UniformFill(0,1) };

double** Matrix::allocateMatrix(FillFunction* fillFunction, int height, int width) {
	double** matrix = (double**)malloc(height * sizeof(double*));
	for (int i = 0; i < height; i++) {
		matrix[i] = (double*)malloc(width * sizeof(double));
		for (int j = 0; j < width; j++) {
			matrix[i][j] = fillFunction->get();
		}
	}
	return matrix;
}

void Matrix::deallocateMatrix(double** A, int height, int width) {
	for (int i = 0; i < height; i++) {
		free(A[i]);
	}
	free(A);
}

void Matrix::add(int m, int n, double** A, double** B, double** C, double scalar1, double scalar2) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			C[i][j] = scalar1 * A[i][j] + scalar2 * B[i][j];
		}
	}
}

void Matrix::scale(int m, int n, double** A, double scalar) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			A[i][j] *= scalar;
		}
	}
}

void Matrix::multiplyABC(int m, int n, int p, double** A, double** B, double** C, bool overwrite) {
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

void Matrix::multiplyAtBC(int m, int n, int p, double** A, double** B, double** C, bool overwrite) {
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

void Matrix::multiplyABtC(int m, int n, int p, double** A, double** B, double** C, bool overwrite){
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C[i][j] = 0;
			}
			for (int k = 0; k < n; k++) {
				C[i][j] += A[i][k] * B[j][k];
			}
		}
	}
}

void Matrix::multiplyABCt(int m, int n, int p, double** A, double** B, double** C, bool overwrite) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C[j][i] = 0;
			}
			for (int k = 0; k < n; k++) {
				C[j][i] += A[i][k] * B[k][j];
			}
		}
	}
}

void Matrix::multiplyAtBtC(int m, int n, int p, double** A, double** B, double** C, bool overwrite) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C[i][j] = 0;
			}
			for (int k = 0; k < n; k++) {
				C[i][j] += A[k][i] * B[j][k];
			}
		}
	}
}

void Matrix::matrixTensorMultiply(int m, int n, int p, double** A, double*** B, double** C, bool overwrite) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C[i][j] = 0;
			}
			for (int k = 0; k < n; k++) {
				C[i][j] += A[i][k] * B[i][k][j];
			}
		}
	}
}

void Matrix::elementMultiply(int m, int n, double** A, double** B, double** C, bool overwrite) {
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

void Matrix::fill(FillFunction* fillFunction, int m, int n, double** A) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			A[i][j] = fillFunction->get();
		}
	}
}

Matrix::ConstantFill::ConstantFill(double value){
	this->value = value;
}

double Matrix::ConstantFill::get() {
	return value;
}

Matrix::NormalFill::NormalFill(double mean, double stdDeviation) {
	distribution = { new normal_distribution<double>(mean, stdDeviation) };
}

double Matrix::NormalFill::get() {
	return (*distribution)(generator);
}

Matrix::UniformFill::UniformFill(double lowerBound, double upperBound){
	distribution = { new uniform_real_distribution<double>(lowerBound, upperBound) };
}

double Matrix::UniformFill::get() {
	return (*distribution)(generator);
}