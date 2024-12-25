#include "MultiHeadAttention.h"

MultiHeadAttention::MultiHeadAttention(int numHeads, int keySize, int valueSize) {
	this->numHeads = numHeads;
	this->keySize = keySize;
	this->valueSize = valueSize;
}

void MultiHeadAttention::propagateLayer(int num) {
	double scalar = 1.0 / sqrt(keySize);
	for (int i = 0; i < numHeads; i++) {
		Matrix2::multiplyABtC(numTokens, prevSize, keySize, prevLayer->neurons[num], Wq[i], Q[num][i], true);
		Matrix2::multiplyABtC(numTokens, prevSize, keySize, prevLayer->neurons[num], Wk[i], K[num][i], true);
		Matrix2::multiplyABtC(numTokens, keySize, numTokens, Q[num][i], K[num][i], A[num][i], true);
		A[num][i].scale(numTokens, numTokens, scalar);
		softmax->operate(numTokens, numTokens, A[num][i], Ao[num][i]);
		Matrix2::multiplyABtC(valueSize, prevSize, numTokens, prevLayer->neurons[num], Wv[i], V[num][i], true);
		Matrix2::multiplyABtC(numTokens, numTokens, valueSize, Ao[num][i], V[num][i], AcSub[num][i], true);
	}
	Ac[num].calculateMatrix(numTokens, numHeads * valueSize);
	Matrix2::multiplyABtC(numTokens, numHeads * valueSize, size, Ac[num], Wo, neurons[num], true);
}

void MultiHeadAttention::backPropagate(int num) {
	Matrix2::multiplyABC(numTokens, size, numHeads * valueSize, neuronGradient[num], Wo, AcGrad[num], true);
	Matrix2::multiplyAtBC(size, numTokens, numHeads * valueSize, neuronGradient[num], Ac[num], WoGrad[num], true);
	outputOptimizer->addGradient(WoGrad[num]);
	prevLayer->neuronGradient[num].fill(Matrix2::ZERO_FILL, numTokens, prevSize);
	float scalar = 1.0 / sqrt(keySize);
	for (int i = 0; i < numHeads; i++) {
		Matrix2::multiplyABC(numTokens, valueSize, numTokens, AcSubGrad[num][i], V[num][i], AoGrad[num][i], true);
		Matrix2::multiplyAtBC(valueSize, numTokens, numTokens, AcSubGrad[num][i], Ao[num][i], VGrad[num][i], true);
		Matrix2::multiplyABC(valueSize, numTokens, prevSize, VGrad[num][i], prevLayer->neurons[num], WvGrad[num][i], true);
		valueOptimizers[i]->addGradient(WvGrad[num][i]);
		Matrix2::multiplyAtBC(numTokens, valueSize, prevSize, VGrad[num][i], Wv[i], prevLayer->neuronGradient[num], false);

		softmax->differentiate(numTokens, numTokens, A[num][i], Ao[num][i], activationGradients[num][i]);
		Matrix3D::matrixTensorMultiply(numTokens, numTokens, numTokens, AoGrad[num][i], activationGradients[num][i], AGrad[num][i], true);
		Matrix2::multiplyABC(numTokens, numTokens, keySize, AGrad[num][i], K[num][i], QGrad[num][i], true);
		QGrad[num][i].scale(numTokens, keySize, scalar);
		Matrix2::multiplyAtBC(numTokens, numTokens, keySize, AGrad[num][i], Q[num][i], KGrad[num][i], true);
		KGrad[num][i].scale(numTokens, keySize, scalar);

		Matrix2::multiplyABC(numTokens, keySize, prevSize, QGrad[num][i], Wq[i], prevLayer->neuronGradient[num], false);
		Matrix2::multiplyAtBC(keySize, numTokens, prevSize, QGrad[num][i], prevLayer->neurons[num], WqGrad[num][i], true);
		queryOptimizers[i]->addGradient(WqGrad[num][i]);
		Matrix2::multiplyABC(numTokens, keySize, prevSize, KGrad[num][i], Wk[i], prevLayer->neuronGradient[num], false);
		Matrix2::multiplyAtBC(keySize, numTokens, prevSize, KGrad[num][i], prevLayer->neuronGradient[num], WkGrad[num][i], true);
		keyOptimizers[i]->addGradient(WkGrad[num][i]);
		prevLayer->backPropagate(num);
	}
}

void MultiHeadAttention::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D*>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
	size = prevLayer->size;
	prevSize = prevLayer->size + 1;
	float std = 1.0 / size;
	Matrix2::NormalFill* normal = new Matrix2::NormalFill(0, std);
	Wq = Matrix2::allocateMatrixArray(normal, numHeads, keySize, prevSize, true);
	Wk = Matrix2::allocateMatrixArray(normal, numHeads, keySize, prevSize, true);
	Wv = Matrix2::allocateMatrixArray(normal, numHeads, valueSize, prevSize, true);
	Wo = Matrix2(normal, size, numHeads * valueSize, true);
}

void MultiHeadAttention::setBatchSize(int batchSize) {
	Layer2D::setBatchSize(batchSize);
	WqGrad = Matrix2::allocateMatrixArray2D(Matrix2::ZERO_FILL, batchSize, numHeads, keySize, prevSize, false);
	WkGrad = Matrix2::allocateMatrixArray2D(Matrix2::ZERO_FILL, batchSize, numHeads, keySize, prevSize, false);
	WvGrad = Matrix2::allocateMatrixArray2D(Matrix2::ZERO_FILL, batchSize, numHeads, valueSize, prevSize, false);
	WoGrad = Matrix2::allocateMatrixArray(Matrix2::ZERO_FILL, batchSize, size, numHeads * valueSize, false);
	K = Matrix2::allocateMatrixArray2D(Matrix2::ZERO_FILL, batchSize, numHeads, maxNumTokens, keySize, false);
	KGrad = Matrix2::allocateMatrixArray2D(Matrix2::ZERO_FILL, batchSize, numHeads, maxNumTokens, keySize, false);
	Q = Matrix2::allocateMatrixArray2D(Matrix2::ZERO_FILL, batchSize, numHeads, maxNumTokens, keySize, false);
	QGrad = Matrix2::allocateMatrixArray2D(Matrix2::ZERO_FILL, batchSize, numHeads, maxNumTokens, keySize, false);
	V = Matrix2::allocateMatrixArray2D(Matrix2::ZERO_FILL, batchSize, numHeads, maxNumTokens, valueSize, false);
	VGrad = Matrix2::allocateMatrixArray2D(Matrix2::ZERO_FILL, batchSize, numHeads, maxNumTokens, valueSize, false);
	A = Matrix2::allocateMatrixArray2D(Matrix2::ZERO_FILL, batchSize, numHeads, maxNumTokens, maxNumTokens, false);
	AGrad = Matrix2::allocateMatrixArray2D(Matrix2::ZERO_FILL, batchSize, numHeads, maxNumTokens, maxNumTokens, false);
	Ao = Matrix2::allocateMatrixArray2D(Matrix2::ZERO_FILL, batchSize, numHeads, maxNumTokens, maxNumTokens, false);
	AoGrad = Matrix2::allocateMatrixArray2D(Matrix2::ZERO_FILL, batchSize, numHeads, maxNumTokens, maxNumTokens, false);
	activationGradients = Matrix3D::allocateMatrix3DArray2D(Matrix2::ZERO_FILL, batchSize, numHeads, maxNumTokens, maxNumTokens, maxNumTokens);
	Ac = Matrix2::allocateMatrixArray(Matrix2::ZERO_FILL, batchSize, maxNumTokens, numHeads * valueSize, false);
	AcGrad = Matrix2::allocateMatrixArray(Matrix2::ZERO_FILL, batchSize, maxNumTokens, numHeads * valueSize, false);
	AcSub = new Matrix2 * [batchSize];
	AcSubGrad = new Matrix2 * [batchSize];
	for (int i = 0; i < batchSize; i++) {
		AcSub[i] = new Matrix2[numHeads];
		AcSubGrad[i] = new Matrix2[numHeads];
		for (int j = 0; j < numHeads; j++) {
			AcSub[i][j] = Ac[i].subMatrix(0, j * valueSize, maxNumTokens, valueSize);
			AcSubGrad[i][j] = AcGrad[i].subMatrix(0, j * valueSize, maxNumTokens, valueSize);
		}
	}
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void MultiHeadAttention::save(ofstream& file) {
	file << "MultiHeadAttention,";
	file << numHeads << "," << keySize << "," << valueSize << ",\n";
	for (int i = 0; i < numHeads; i++) {
		for (int j = 0; j < keySize; j++) {
			for (int k = 0; k < prevSize; k++) {
				file << Wq[i](j,k) << ",";
			}
			file << "\n";
		}
		for (int j = 0; j < keySize; j++) {
			for (int k = 0; k < prevSize; k++) {
				file << Wk[i](j,k) << ",";
			}
			file << "\n";
		}
		for (int j = 0; j < valueSize; j++) {
			for (int k = 0; k < prevSize; k++) {
				file << Wv[i](j,k) << ",";
			}
			file << "\n";
		}
	}
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < numHeads * valueSize; j++) {
			file << Wo(i,j) << ",";
		}
		file << "\n";
	}
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void MultiHeadAttention::applyGradients(float learningRate, int t) {
	outputOptimizer->applyGradient(Wo, t, learningRate, batchSize);
	for (int i = 0; i < numHeads; i++) {
		queryOptimizers[i]->applyGradient(Wq[i], t, learningRate, batchSize);
		keyOptimizers[i]->applyGradient(Wk[i], t, learningRate, batchSize);
		valueOptimizers[i]->applyGradient(Wv[i], t, learningRate, batchSize);
	}
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void MultiHeadAttention::setOptimizer(Optimizer* optimizer) {
	outputOptimizer = optimizer->clone();
	outputOptimizer->setDimensions(size, numHeads * valueSize);
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

int MultiHeadAttention::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	current += size * numHeads * valueSize;
	current += numHeads * valueSize * prevSize;
	current += 2 * numHeads * keySize * prevSize;
	return current;
}