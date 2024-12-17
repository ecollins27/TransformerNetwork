#include "MatrixBatch.h"



MatrixBatch::MatrixBatch(bool saveTranspose) {
	this->saveTranspose = saveTranspose;
}

float& MatrixBatch::operator()(int i, int j, int k) {
	return matrices[i](i, j);
}

void MatrixBatch::setBatchSize(Matrix2::FillFunction* fillFunction, int batchSize, int height, int width) {
	matrices = new Matrix2[batchSize];
	for (int i = 0; i < batchSize; i++) {
		matrices[i] = Matrix2(fillFunction, height, width, saveTranspose);
	}
}