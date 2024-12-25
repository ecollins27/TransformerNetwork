#include "MatrixBatch.h"



MatrixBatch::MatrixBatch(Matrix2::FillFunction* fillFunction, int batchSize, int height, int width, bool saveTranspose) {
	this->saveTranspose = saveTranspose;
	matrices = new Matrix2[batchSize];
	for (int i = 0; i < batchSize; i++) {
		matrices[i] = Matrix2(fillFunction, height, width, saveTranspose);
	}
}

MatrixBatch::MatrixBatch(int batchSize) {
	this->batchSize = batchSize;
	matrices = new Matrix2[batchSize];
}

float& MatrixBatch::operator()(int i, int j, int k) {
	return matrices[i](i, j);
}

Matrix2& MatrixBatch::operator[](int i) {
	return matrices[i];
}