#include "Matrix3D.h"


Matrix3D::Matrix3D(Matrix2::FillFunction* fillFunction, int depth, int height, int width) {
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

void Matrix3D::matrixTensorMultiply(int m, int n, int p, Matrix2 A, Matrix3D B, Matrix2 C, bool overwrite) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (overwrite) {
				C(i, j) = Matrix2::dotProduct(n, A.matrix[i], B.matrix[i][j]);
			}
			else {
				C(i, j) += Matrix2::dotProduct(n, A.matrix[i], B.matrix[i][j]);
			}
		}
	}
}