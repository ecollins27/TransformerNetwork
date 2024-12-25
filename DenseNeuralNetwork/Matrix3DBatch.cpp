#include "Matrix3DBatch.h"

Matrix3DBatch::Matrix3DBatch(Matrix2::FillFunction* fillFunction, int batchSize, int depth, int height, int width) {
	matrices = new Matrix3D[batchSize];
	for (int i = 0; i < batchSize; i++) {
		matrices[i] = Matrix3D(fillFunction, depth, height, width);
	}
}

float& Matrix3DBatch::operator()(int i, int j, int k, int l) {
	return matrices[i](j, k, l);
}

Matrix3D& Matrix3DBatch::operator[](int i) {
	return matrices[i];
}