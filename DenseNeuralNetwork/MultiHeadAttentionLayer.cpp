#include "MultiHeadAttentionLayer.h"

MultiHeadAttentionLayer::MultiHeadAttentionLayer(int numHeads, int keySize, int valueSize) {
	this->numHeads = numHeads;
	this->keySize = keySize;
	this->valueSize = valueSize;
}

void MultiHeadAttentionLayer::propagateLayer() {
	double scalar = 1.0 / sqrt(keySize);
	for (int i = 0; i < numHeads; i++) {
		Matrix::matrixMultiplyABtC(batchSize, prevSize, keySize, prevLayer->neurons, Wq[i], Q[i], true);
		Matrix::transpose(batchSize, keySize, Q[i], QTrans[i]);
		Matrix::matrixMultiplyABtC(batchSize, prevSize, keySize, prevLayer->neurons, Wk[i], K[i], true);
		Matrix::transpose(batchSize, keySize, K[i], KTrans[i]);
		Matrix::matrixMultiplyABtC(batchSize, keySize, batchSize, Q[i], K[i], A[i], true);
		Matrix::scale(batchSize, batchSize, A[i], scalar);
		softmax->operate(batchSize, batchSize, A[i], Ao[i]);
		Matrix::transpose(batchSize, batchSize, Ao[i], AoTrans[i]);
		Matrix::matrixMultiplyABtC(valueSize, prevSize, batchSize, Wv[i], prevLayer->neurons, V[i], true);
		Matrix::transpose(valueSize, batchSize, V[i], VTrans[i]);
		Matrix::matrixMultiplyABtC(valueSize, batchSize, batchSize, V[i], Ao[i], &AcTrans[i * valueSize], true);
	}
	Matrix::transpose(numHeads * valueSize, batchSize, AcTrans, Ac);
	Matrix::matrixMultiplyABtC(batchSize, numHeads * valueSize, size, Ac, Wo, neurons, true);
	Matrix::transpose(batchSize, size, neurons, neuronsTranspose);
}

void MultiHeadAttentionLayer::backPropagate() {
	Matrix::transpose(batchSize, size, neuronGradient, neuronGradientTrans);
	Matrix::matrixMultiplyABtC(batchSize, size, numHeads * valueSize, neuronGradient, WoTrans, AcGrad, true);
	Matrix::transpose(batchSize, numHeads * valueSize, AcGrad, AcGradTrans);
	Matrix::matrixMultiplyABtC(size, batchSize, numHeads * valueSize, neuronGradientTrans, AcTrans, WoGrad, true);
	Matrix::fill(Matrix::ZERO_FILL, batchSize, prevSize, prevLayer->neuronGradient);
	float scalar = 1.0 / sqrt(keySize);
	for (int i = 0; i < numHeads; i++) {
		Matrix::subMatrixMultiplyABtC(batchSize, valueSize, batchSize, AcGrad, VTrans[i], AoGrad[i], true, i * valueSize);
		Matrix::matrixMultiplyABtC(valueSize, batchSize, batchSize, &AcGradTrans[i * valueSize], AoTrans[i], VGrad[i], true);
		Matrix::transpose(valueSize, batchSize, VGrad[i], VGradTrans[i]);
		Matrix::matrixMultiplyABtC(valueSize, batchSize, prevSize, VGrad[i], prevLayer->neuronsTranspose, WvGrad[i], true);
		Matrix::matrixMultiplyABtC(batchSize, valueSize, prevSize, VGradTrans[i], WvTrans[i], prevLayer->neuronGradient, false);
		softmax->differentiate(batchSize, batchSize, A[i], Ao[i], activationGradients[i]);
		Matrix::matrixTensorMultiply(batchSize, batchSize, batchSize, AoGrad[i], activationGradients[i], AGrad[i], true);
		Matrix::transpose(batchSize, batchSize, AGrad[i], AGradTrans[i]);
		Matrix::matrixMultiplyABtC(batchSize, batchSize, keySize, AGrad[i], KTrans[i], QGrad[i], true);
		Matrix::scale(batchSize, keySize, QGrad[i], scalar);
		Matrix::transpose(batchSize, keySize, QGrad[i], QGradTrans[i]);
		Matrix::matrixMultiplyABtC(batchSize, batchSize, keySize, AGradTrans[i], QTrans[i], KGrad[i], true);
		Matrix::scale(batchSize, keySize, KGrad[i], scalar);
		Matrix::transpose(batchSize, keySize, KGrad[i], KGradTrans[i]);
		Matrix::matrixMultiplyABtC(batchSize, keySize, prevSize, QGrad[i], WqTrans[i], prevLayer->neuronGradient, false);
		Matrix::matrixMultiplyABtC(keySize, batchSize, prevSize, QGradTrans[i], prevLayer->neuronsTranspose, WqGrad[i], true);
		Matrix::matrixMultiplyABtC(batchSize, keySize, prevSize, KGrad[i], WkTrans[i], prevLayer->neuronGradient, false);
		Matrix::matrixMultiplyABtC(keySize, batchSize, prevSize, KGradTrans[i], prevLayer->neuronsTranspose, WkGrad[i], true);
	}
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
}

void MultiHeadAttentionLayer::setPrevLayer(Layer* prevLayer) {
	this->prevLayer = prevLayer;
	this->size = prevLayer->size;
	this->prevSize = size + 1;
	float std = 1.0 / size;
	Matrix::FillFunction* fillFunction = { new Matrix::NormalFill(0, std) };
	Wq = Matrix::allocate3DMatrix(fillFunction, numHeads, keySize, prevSize);
	WqTrans = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, prevSize, keySize);
	WqGrad = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, keySize, prevSize);
	Wk = Matrix::allocate3DMatrix(fillFunction, numHeads, keySize, prevSize);
	WkTrans = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, prevSize, keySize);
	WkGrad = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, keySize, prevSize);
	Wv = Matrix::allocate3DMatrix(fillFunction, numHeads, valueSize, prevSize);
	WvTrans = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, prevSize, valueSize);
	WvGrad = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, valueSize, prevSize);
	Wo = Matrix::allocateMatrix(fillFunction, size, numHeads * valueSize);
	WoTrans = Matrix::allocateMatrix(Matrix::ZERO_FILL, numHeads * valueSize, size);
	Matrix::transpose(size, numHeads * valueSize, Wo, WoTrans);
	WoGrad = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, numHeads * valueSize);
	for (int i = 0; i < numHeads; i++) {
		Matrix::transpose(keySize, prevSize, Wq[i], WqTrans[i]);
		Matrix::transpose(keySize, prevSize, Wk[i], WkTrans[i]);
		Matrix::transpose(valueSize, prevSize, Wv[i], WvTrans[i]);
	}
}

void MultiHeadAttentionLayer::setBatchSize(int batchSize) {
	this->batchSize = batchSize;
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void MultiHeadAttentionLayer::applyGradients(float learningRate, int t) {
	optimizer->applyGradient(WoGrad, Wo, t, learningRate);
	Matrix::transpose(size, numHeads * valueSize, Wo, WoTrans);
	for (int i = 0; i < numHeads; i++) {
		queryOptimizers[i]->applyGradient(WqGrad[i], Wq[i], t, learningRate);
		Matrix::transpose(keySize, prevSize, Wq[i], WqTrans[i]);
		keyOptimizers[i]->applyGradient(WkGrad[i], Wk[i], t, learningRate);
		Matrix::transpose(keySize, prevSize, Wk[i], WkTrans[i]);
		valueOptimizers[i]->applyGradient(WvGrad[i], Wv[i], t, learningRate);
		Matrix::transpose(valueSize, prevSize, Wv[i], WvTrans[i]);
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
	current += size * numHeads * valueSize;
	current += numHeads * valueSize * prevSize;
	current += 2 * numHeads * keySize * prevSize;
	return current;
}

void MultiHeadAttentionLayer::setMaxBatchSize(int maxBatchSize) {
	if (Q != NULL) {
		Matrix::deallocate3DMatrix(Q, numHeads, this->maxBatchSize, keySize);
		Matrix::deallocate3DMatrix(QTrans, numHeads, keySize, this->maxBatchSize);
		Matrix::deallocate3DMatrix(QGrad, numHeads, this->maxBatchSize, keySize);
		Matrix::deallocate3DMatrix(QGradTrans, numHeads, keySize, this->maxBatchSize);
		Matrix::deallocate3DMatrix(K, numHeads, this->maxBatchSize, keySize);
		Matrix::deallocate3DMatrix(KTrans, numHeads, keySize, this->maxBatchSize);
		Matrix::deallocate3DMatrix(KGrad, numHeads, this->maxBatchSize, keySize);
		Matrix::deallocate3DMatrix(KGradTrans, numHeads, keySize, this->maxBatchSize);
		Matrix::deallocate3DMatrix(V, numHeads, valueSize, this->maxBatchSize);
		Matrix::deallocate3DMatrix(VTrans, numHeads, this->maxBatchSize, valueSize);
		Matrix::deallocate3DMatrix(VGrad, numHeads, valueSize, this->maxBatchSize);
		Matrix::deallocate3DMatrix(VGradTrans, numHeads, this->maxBatchSize, valueSize);
		Matrix::deallocate3DMatrix(A, numHeads, this->maxBatchSize, this->maxBatchSize);
		Matrix::deallocate3DMatrix(AGrad, numHeads, this->maxBatchSize, this->maxBatchSize);
		Matrix::deallocate3DMatrix(AGradTrans, numHeads, this->maxBatchSize, this->maxBatchSize);
		Matrix::deallocate3DMatrix(Ao, numHeads, this->maxBatchSize, this->maxBatchSize);
		Matrix::deallocate3DMatrix(AoTrans, numHeads, this->maxBatchSize, this->maxBatchSize);
		Matrix::deallocate3DMatrix(AoGrad, numHeads, this->maxBatchSize, this->maxBatchSize);
		Matrix::deallocateMatrix(Ac, this->maxBatchSize, numHeads * valueSize);
		Matrix::deallocateMatrix(AcTrans, numHeads * valueSize, this->maxBatchSize);
		Matrix::deallocateMatrix(AcGrad, this->maxBatchSize, numHeads * valueSize);
		Matrix::deallocateMatrix(AcGradTrans, numHeads * valueSize, this->maxBatchSize);
		Matrix::deallocate4DMatrix(activationGradients, numHeads, this->maxBatchSize, this->maxBatchSize, this->maxBatchSize);
		Matrix::deallocateMatrix(neurons, this->maxBatchSize, size + 1);
		Matrix::deallocateMatrix(neuronsTranspose, size + 1, this->maxBatchSize);
		Matrix::deallocateMatrix(neuronGradient, this->maxBatchSize, size + 1);
		Matrix::deallocateMatrix(neuronGradientTrans, size + 1, this->maxBatchSize);
	}
	this->maxBatchSize = maxBatchSize;
	Q = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, keySize);
	QTrans = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, keySize, maxBatchSize);
	QGrad = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, keySize);
	QGradTrans = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, keySize, maxBatchSize);
	K = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, keySize);
	KTrans = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, keySize, maxBatchSize);
	KGrad = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, keySize);
	KGradTrans = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, keySize, maxBatchSize);
	V = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, valueSize, maxBatchSize);
	VTrans = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, valueSize);
	VGrad = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, valueSize, maxBatchSize);
	VGradTrans = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, valueSize);
	A = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, maxBatchSize);
	AGrad = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, maxBatchSize);
	AGradTrans = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, maxBatchSize);
	Ao = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, maxBatchSize);
	AoTrans = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, maxBatchSize);
	AoGrad = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, maxBatchSize);
	Ac = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, numHeads * valueSize);
	AcTrans = Matrix::allocateMatrix(Matrix::ZERO_FILL, numHeads * valueSize, maxBatchSize);
	AcGrad = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, numHeads * valueSize);
	AcGradTrans = Matrix::allocateMatrix(Matrix::ZERO_FILL, numHeads * valueSize, maxBatchSize);
	activationGradients = Matrix::allocate4DMatrix(Matrix::ZERO_FILL, numHeads, maxBatchSize, maxBatchSize, maxBatchSize);
	neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size + 1);
	neuronsTranspose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, maxBatchSize);
	neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size + 1);
	neuronGradientTrans = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, maxBatchSize);
	for (int i = 0; i < maxBatchSize; i++) {
		neurons[i][size] = 1;
		neuronsTranspose[size][i] = 1;
		neuronGradient[i][size] = 0;
		neuronGradientTrans[size][i] = 0;
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