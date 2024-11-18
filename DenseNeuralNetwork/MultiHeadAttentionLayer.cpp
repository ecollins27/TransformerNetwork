#include "MultiHeadAttentionLayer.h"

MultiHeadAttentionLayer::MultiHeadAttentionLayer(int numHeads, int keySize, int valueSize) {
	this->numHeads = numHeads;
	this->keySize = keySize;
	this->valueSize = valueSize;
}

void MultiHeadAttentionLayer::predict() {
	double scalar = 1.0 / sqrt(keySize);
	for (int i = 0; i < numHeads; i++) {
		Matrix::matrixMultiplyABtC(batchSize, prevSize, keySize, prevLayer->neurons, Wq[i], Q[i], true);
		Matrix::matrixMultiplyABtC(batchSize, prevSize, keySize, prevLayer->neurons, Wk[i], K[i], true);
		Matrix::matrixMultiplyABtC(batchSize, keySize, batchSize, Q[i], K[i], A[i], true);
		Matrix::scale(batchSize, batchSize, A[i], scalar);
		softmax->operate(batchSize, batchSize, A[i], Ao[i]);
		Matrix::matrixMultiplyABtC(valueSize, prevSize, batchSize, Wv[i], prevLayer->neuronsTranspose, V[i], true);
		Matrix::matrixMultiplyABtC(valueSize, batchSize, batchSize, V[i], Ao[i], &AcTrans[i * valueSize], true);
	}
	Matrix::transpose(numHeads * valueSize, batchSize, AcTrans, Ac);
	Matrix::matrixMultiplyABtC(batchSize, numHeads * valueSize, size, Ac, Wo, neurons, true);
	Matrix::transpose(batchSize, size, neurons, neuronsTranspose);
	if (nextLayer != NULL) {
		nextLayer->predict();
	}
}

void MultiHeadAttentionLayer::forwardPropagate() {
	double scalar = 1.0 / sqrt(keySize);
	for (int i = 0; i < numHeads; i++) {
		Matrix::matrixMultiplyABtC(batchSize, prevSize, keySize, prevLayer->neurons, Wq[i], Q[i], true);
		Matrix::matrixMultiplyABtC(batchSize, prevSize, keySize, prevLayer->neurons, Wk[i], K[i], true);
		Matrix::matrixMultiplyABtC(batchSize, keySize, batchSize, Q[i], K[i], A[i], true);
		Matrix::scale(batchSize, batchSize, A[i], scalar);
		softmax->operate(batchSize, batchSize, A[i], Ao[i]);
		Matrix::matrixMultiplyABtC(valueSize, prevSize, batchSize, Wv[i], prevLayer->neuronsTranspose, V[i], true);
		Matrix::matrixMultiplyABtC(valueSize, batchSize, batchSize, V[i], Ao[i], &AcTrans[i * valueSize], true);
	}
	Matrix::transpose(numHeads * valueSize, batchSize, AcTrans, Ac);
	Matrix::matrixMultiplyABtC(batchSize, numHeads * valueSize, size, Ac, Wo, neurons, true);
	Matrix::transpose(batchSize, size, neurons, neuronsTranspose);
	if (nextLayer != NULL) {
		nextLayer->forwardPropagate();
	}
}

void MultiHeadAttentionLayer::backPropagate() {
	Matrix::matrixMultiplyABC(batchSize, size, numHeads * valueSize, neuronGradient, Wo, AcGrad, true);
	Matrix::transpose(batchSize, numHeads * valueSize, AcGrad, AcGradTrans);
	Matrix::matrixMultiplyAtBC(size, batchSize, numHeads * valueSize, neuronGradient, Ac, WoGrad, true);
	Matrix::fill(Matrix::ZERO_FILL, batchSize, size, prevLayer->neuronGradient);
	float scalar = 1.0 / sqrt(keySize);
	for (int i = 0; i < numHeads; i++) {
		Matrix::matrixMultiplyAtBC(batchSize, valueSize, batchSize, &AcGradTrans[i * valueSize], V[i], AoGrad[i], true);
		Matrix::matrixMultiplyABC(valueSize, batchSize, batchSize, &AcGradTrans[i * valueSize], Ao[i], VGrad[i], true);
		Matrix::matrixMultiplyABC(valueSize, batchSize, prevSize, VGrad[i], prevLayer->neurons, WvGrad[i], true);
		Matrix::matrixMultiplyAtBC(batchSize, valueSize, prevSize, VGrad[i], Wv[i], prevLayer->neuronGradient, false);
		softmax->differentiate(batchSize, prevSize, A[i], Ao[i], activationGradients[i]);
		Matrix::matrixTensorMultiply(batchSize, batchSize, batchSize, AoGrad[i], activationGradients[i], AGrad[i], true);
		Matrix::matrixMultiplyABC(batchSize, batchSize, keySize, AGrad[i], K[i], QGrad[i], true);
		Matrix::scale(batchSize, keySize, QGrad[i], scalar);
		Matrix::matrixMultiplyAtBC(batchSize, batchSize, keySize, AGrad[i], Q[i], KGrad[i], true);
		Matrix::scale(batchSize, keySize, KGrad[i], scalar);
		Matrix::matrixMultiplyABC(batchSize, keySize, prevSize, QGrad[i], Wq[i], prevLayer->neuronGradient, false);
		Matrix::matrixMultiplyAtBC(keySize, batchSize, prevSize, QGrad[i], prevLayer->neurons, WqGrad[i], true);
		Matrix::matrixMultiplyABC(batchSize, keySize, prevSize, KGrad[i], Wk[i], prevLayer->neuronGradient, false);
		Matrix::matrixMultiplyAtBC(keySize, batchSize, prevSize, KGrad[i], prevLayer->neurons, WkGrad[i], true);
	}
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
}

void MultiHeadAttentionLayer::setPrevLayer(Layer* prevLayer) {
	this->prevLayer = prevLayer;
	this->prevSize = prevLayer->size + 1;
	this->size = prevSize;
	float std = 1.0 / size;
	Matrix::FillFunction* fillFunction = { new Matrix::NormalFill(0, std) };
	Wq = Matrix::allocate3DMatrix(fillFunction, numHeads, keySize, prevSize);
	WqGrad = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, keySize, prevSize);
	Wk = Matrix::allocate3DMatrix(fillFunction, numHeads, keySize, prevSize);
	WkGrad = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, keySize, prevSize);
	Wv = Matrix::allocate3DMatrix(fillFunction, numHeads, valueSize, prevSize);
	WvGrad = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, valueSize, prevSize);
	Wo = Matrix::allocateMatrix(fillFunction, size, numHeads * valueSize);
	WoGrad = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, numHeads * valueSize);
}
void MultiHeadAttentionLayer::setNextLayer(Layer* nextLayer) {
	this->nextLayer = nextLayer;
}

void MultiHeadAttentionLayer::setBatchSize(int batchSize) {
	this->batchSize = batchSize;
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void MultiHeadAttentionLayer::applyGradients(float learningRate, int t) {
	optimizer->applyGradient(WoGrad, Wo, t, learningRate);
	for (int i = 0; i < numHeads; i++) {
		queryOptimizers[i]->applyGradient(WqGrad[i], Wq[i], t, learningRate);
		keyOptimizers[i]->applyGradient(WkGrad[i], Wk[i], t, learningRate);
		valueOptimizers[i]->applyGradient(WvGrad[i], Wv[i], t, learningRate);
	}
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void MultiHeadAttentionLayer::setOptimizer(Optimizer* optimizer) {
	this->optimizer = optimizer->clone();
	this->optimizer->setDimensions(size, numHeads * valueSize);
	queryOptimizers = (Optimizer**)malloc(numHeads * sizeof(Optimizer*));
	keyOptimizers = (Optimizer**)malloc(numHeads * sizeof(Optimizer*));
	valueOptimizers = (Optimizer**)malloc(numHeads * sizeof(Optimizer*));
	for (int i = 0; i < numHeads; i++) {
		queryOptimizers[i] = optimizer->clone();
		queryOptimizers[i]->setDimensions(keySize, prevSize);
		keyOptimizers[i] = optimizer->clone();
		keyOptimizers[i]->setDimensions(keySize, prevSize);
		valueOptimizers[i] = optimizer->clone();
		valueOptimizers[i]->setDimensions(valueSize, prevSize);
	}
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

int MultiHeadAttentionLayer::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	return current + 2 * numHeads * valueSize * size + 2 * numHeads * keySize * size;
}

void MultiHeadAttentionLayer::setMaxBatchSize(int maxBatchSize) {
	if (Q != NULL) {
		Matrix::deallocate3DMatrix(Q, numHeads, this->maxBatchSize, keySize);
		Matrix::deallocate3DMatrix(QGrad, numHeads, this->maxBatchSize, keySize);
		Matrix::deallocate3DMatrix(K, numHeads, this->maxBatchSize, keySize);
		Matrix::deallocate3DMatrix(KGrad, numHeads, this->maxBatchSize, keySize);
		Matrix::deallocate3DMatrix(V, numHeads, valueSize, this->maxBatchSize);
		Matrix::deallocate3DMatrix(VGrad, numHeads, valueSize, this->maxBatchSize);
		Matrix::deallocate3DMatrix(A, numHeads, this->maxBatchSize, this->maxBatchSize);
		Matrix::deallocate3DMatrix(AGrad, numHeads, this->maxBatchSize, this->maxBatchSize);
		Matrix::deallocate3DMatrix(Ao, numHeads, this->maxBatchSize, this->maxBatchSize);
		Matrix::deallocate3DMatrix(AoGrad, numHeads, this->maxBatchSize, this->maxBatchSize);
		Matrix::deallocateMatrix(Ac, this->maxBatchSize, numHeads * valueSize);
		Matrix::deallocateMatrix(AcTrans, numHeads * valueSize, this->maxBatchSize);
		Matrix::deallocateMatrix(AcGrad, this->maxBatchSize, numHeads * valueSize);
		Matrix::deallocateMatrix(AcGradTrans, numHeads * valueSize, this->maxBatchSize);
		Matrix::deallocate4DMatrix(activationGradients, numHeads, this->maxBatchSize, this->maxBatchSize, this->maxBatchSize);
		Matrix::deallocateMatrix(neurons, this->maxBatchSize, size);
		Matrix::deallocateMatrix(neuronsTranspose, size, this->maxBatchSize);
		Matrix::deallocateMatrix(neuronGradient, this->maxBatchSize, size);
	}
	this->maxBatchSize = maxBatchSize;
	Q = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, keySize);
	QGrad = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, keySize);
	K = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, keySize);
	KGrad = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, keySize);
	V = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, valueSize, maxBatchSize);
	VGrad = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, valueSize, maxBatchSize);
	A = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, maxBatchSize);
	AGrad = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, maxBatchSize);
	Ao = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, maxBatchSize);
	AoGrad = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, maxBatchSize);
	Ac = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, numHeads * valueSize);
	AcTrans = Matrix::allocateMatrix(Matrix::ZERO_FILL, numHeads * valueSize, maxBatchSize);
	AcGrad = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, numHeads * valueSize);
	AcGradTrans = Matrix::allocateMatrix(Matrix::ZERO_FILL, numHeads * valueSize, maxBatchSize);
	activationGradients = Matrix::allocate4DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, maxBatchSize, maxBatchSize);
	neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size + 1);
	neuronsTranspose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, maxBatchSize);
	neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size + 1);
	for (int i = 0; i < maxBatchSize; i++) {
		neurons[i][size] = 1;
		neuronsTranspose[size][i] = 1;
		neuronGradient[i][size = 1];
	}
	if (nextLayer != NULL) {
		nextLayer->setMaxBatchSize(maxBatchSize);
	}
}

void MultiHeadAttentionLayer::save(ofstream& file) {
	file << "MultiHeadAttention,";
	file << numHeads << "," << keySize << "," << valueSize << ",\n";
	for (int i = 0; i < numHeads; i++) {
		for (int j = 0; j < keySize; j++) {
			for (int k = 0; k < prevSize; k++) {
				file << Wq[i][j][k] << ",";
			}
			file << "\n";
		}
		for (int j = 0; j < keySize; j++) {
			for (int k = 0; k < prevSize; k++) {
				file << Wk[i][j][k] << ",";
			}
			file << "\n";
		}
		for (int j = 0; j < valueSize; j++) {
			for (int k = 0; k < prevSize; k++) {
				file << Wv[i][j][k] << ",";
			}
			file << "\n";
		}
	}
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < numHeads * valueSize; j++) {
			file << Wo[i][j] << ",";
		}
		file << "\n";
	}
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}