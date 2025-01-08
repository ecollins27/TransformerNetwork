#include "Matrix3D.h"


Matrix3D::Matrix3D(Matrix::FillFunction* fillFunction, int depth, int height, int width) {
	matrix = new float** [depth];
	for (int i = 0; i < depth; i++) {
		matrix[i] = new float* [height];
		for (int j = 0; j < height; j++) {
			matrix[i][j] = new float[width];
			for (int k = 0; k < width; k++) {
				matrix[i][j][k] = (*fillFunction)();
			}
		}
	}
}

float& Matrix3D::operator()(int i, int j, int k) {
	return matrix[i][j][k];
}

bool Matrix3D::containsIllegalValue(int depth, int height, int width) {
	for (int i = 0; i < depth; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < width; k++) {
				if (matrix[i][j][k] != matrix[i][j][k] || isinf(matrix[i][j][k])) {
					return true;
				}
			}
		}
	}
	return false;
}

Matrix3D* Matrix3D::allocateMatrix3DArray(Matrix::FillFunction* fillFunction, int x1, int x2, int x3, int x4) {
	Matrix3D* array = new Matrix3D[x1];
	for (int i = 0; i < x1; i++) {
		array[i] = Matrix3D(fillFunction, x2, x3, x4);
	}
	return array;
}

Matrix3D** Matrix3D::allocateMatrix3DArray2D(Matrix::FillFunction* fillFunction, int x1, int x2, int x3, int x4, int x5) {
	Matrix3D** array = new Matrix3D * [x1];
	for (int i = 0; i < x1; i++) {
		array[i] = new Matrix3D[x2];
		for (int j = 0; j < x2; j++) {
			array[i][j] = Matrix3D(fillFunction, x3, x4, x5);
		}
	}
	return array;
}

void Matrix3D::matrixTensorMultiply(int m, int n, int p, Matrix A, Matrix3D B, Matrix C, bool overwrite) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C.r(i, j) = Matrix::dotProduct(n, A.matrix[i], B.matrix[i][j]);
			}
			else {
				C.r(i, j) += Matrix::dotProduct(n, A.matrix[i], B.matrix[i][j]);
			}
		}
	}
}