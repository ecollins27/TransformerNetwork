#include "Matrix3DBatch.h"

float& Matrix3DBatch::operator()(int i, int j, int k, int l) {
	return matrices[i](j, k, l);
}
void Matrix3DBatch::setBatchSize(Matrix2::FillFunction* fillFunction, int batchSize, int depth, int height, int width) {
	matrices = new Matrix3D[batchSize];
	for (int i = 0; i < batchSize; i++) {
		matrices[i] = Matrix3D(fillFunction, depth, height, width);
	}
}