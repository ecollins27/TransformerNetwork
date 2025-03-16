#include "TransformerAttention.h"
#include "Model.h"
#include "ModelParser.h"

const string TransformerAttention::LAYER_NAME = "TransformerAttention";

TransformerAttention::TransformerAttention(int numHeads, int keySize, int valueSize) {
	this->numHeads = numHeads;
	this->keySize = keySize;
	this->valueSize = valueSize;
	softmax = Activation::SOFTMAX->clone();
}

void TransformerAttention::propagateLayer(int num) {
	double scalar = 1.0 / sqrt(keySize);
	for (int i = 0; i < numHeads; i++) {
		Matrix::multiplyABtC(numTokens[num], prevSize, keySize, prevLayer->neurons[num], Wq[i], Q[num][i], true);
		Matrix::multiplyABtC(numTokens[num], prevSize, keySize, prevLayer->neurons[num], Wk[i], K[num][i], true);
		Matrix::multiplyABtC(numTokens[num], keySize, numTokens[num], Q[num][i], K[num][i], A[num][i], true);
		A[num][i].scale(numTokens[num], numTokens[num], scalar);
		softmax->operate(numTokens[num], numTokens[num], A[num][i], A[num][i]);
		Matrix::multiplyABtC(valueSize, prevSize, numTokens[num], Wv[i], prevLayer->neurons[num], V[num][i], true);
		Matrix::multiplyABtC(numTokens[num], numTokens[num], valueSize, A[num][i], V[num][i], AcSub[num][i], true);
	}
	Matrix::multiplyABtC(numTokens[num], numHeads * valueSize, size, Ac[num], Wo, neurons[num], true);
}

void TransformerAttention::backPropagate(int num) {
	Matrix::multiplyABC(numTokens[num], size, numHeads * valueSize, neuronGradient[num], Wo, AcGrad[num], true);
	Matrix::multiplyAtBC(size, numTokens[num], numHeads * valueSize, neuronGradient[num], Ac[num], WoGrad[num], true);
	prevLayer->neuronGradient[num].constantFill(0, numTokens[num], prevSize);
	float scalar = 1.0 / sqrt(keySize);
	for (int i = 0; i < numHeads; i++) {
		Matrix::multiplyABC(numTokens[num], valueSize, numTokens[num], AcSubGrad[num][i], V[num][i], AGrad[num][i], true);
		Matrix::multiplyAtBC(valueSize, numTokens[num], numTokens[num], AcSubGrad[num][i], A[num][i], VGrad[num][i], true);
		Matrix::multiplyABC(valueSize, numTokens[num], prevSize, VGrad[num][i], prevLayer->neurons[num], WvGrad[num][i], true);
		Matrix::multiplyAtBC(numTokens[num], valueSize, prevSize, VGrad[num][i], Wv[i], prevLayer->neuronGradient[num], false);

		softmax->differentiate(numTokens[num], numTokens[num], A[num][i], A[num][i], AGrad[num][i], AGrad[num][i]);
		Matrix::multiplyABC(numTokens[num], numTokens[num], keySize, AGrad[num][i], K[num][i], QGrad[num][i], true);
		QGrad[num][i].scale(numTokens[num], keySize, scalar);
		Matrix::multiplyAtBC(numTokens[num], numTokens[num], keySize, AGrad[num][i], Q[num][i], KGrad[num][i], true);
		KGrad[num][i].scale(numTokens[num], keySize, scalar);

		Matrix::multiplyABC(numTokens[num], keySize, prevSize, QGrad[num][i], Wq[i], prevLayer->neuronGradient[num], false);
		Matrix::multiplyAtBC(keySize, numTokens[num], prevSize, QGrad[num][i], prevLayer->neurons[num], WqGrad[num][i], true);
		Matrix::multiplyABC(numTokens[num], keySize, prevSize, KGrad[num][i], Wk[i], prevLayer->neuronGradient[num], false);
		Matrix::multiplyAtBC(keySize, numTokens[num], prevSize, KGrad[num][i], prevLayer->neuronGradient[num], WkGrad[num][i], true);
	}
	prevLayer->backPropagate(num);
}

void TransformerAttention::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
	size = prevLayer->size;
	prevSize = prevLayer->size + 1;
	float std = 1.0 / size;
	Matrix::NormalFill* normal = new Matrix::NormalFill(0, std);
	Wq = Matrix::allocateMatrixArray(normal, numHeads, keySize, prevSize, true);
	Wk = Matrix::allocateMatrixArray(normal, numHeads, keySize, prevSize, true);
	Wv = Matrix::allocateMatrixArray(normal, numHeads, valueSize, prevSize, true);
	Wo = Matrix(normal, size, numHeads * valueSize, true);
}

void TransformerAttention::setBatchSize(int batchSize) {
	Layer2D::initNeurons(batchSize);
	WqGrad = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, keySize, prevSize, false);
	WkGrad = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, keySize, prevSize, false);
	WvGrad = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, valueSize, prevSize, false);
	WoGrad = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, size, numHeads * valueSize, false);
	K = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, maxNumTokens, keySize, true);
	KGrad = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, maxNumTokens, keySize, true);
	Q = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, maxNumTokens, keySize, true);
	QGrad = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, maxNumTokens, keySize, true);
	V = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, valueSize, maxNumTokens, true);
	VGrad = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, valueSize, maxNumTokens, true);
	A = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, maxNumTokens, maxNumTokens, true);
	AGrad = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, maxNumTokens, maxNumTokens, true);
	Ac = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, maxNumTokens, numHeads * valueSize, true);
	AcGrad = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, maxNumTokens, numHeads * valueSize, true);
	AcSub = new Matrix * [batchSize];
	AcSubGrad = new Matrix * [batchSize];
	for (int i = 0; i < batchSize; i++) {
		AcSub[i] = new Matrix[numHeads];
		AcSubGrad[i] = new Matrix[numHeads];
		for (int j = 0; j < numHeads; j++) {
			AcSub[i][j] = Ac[i].subMatrix(0, j * valueSize, maxNumTokens, valueSize);
			AcSubGrad[i][j] = AcGrad[i].subMatrix(0, j * valueSize, maxNumTokens, valueSize);
		}
	}
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void TransformerAttention::save(ofstream& file) {
	file << LAYER_NAME << ",";
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

void TransformerAttention::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	int numHeads = ModelParser::getNextInt(line, commaIndex, newCommaIndex);
	int keySize = ModelParser::getNextInt(line, commaIndex, newCommaIndex);
	int valueSize = ModelParser::getNextInt(line, commaIndex, newCommaIndex);
	TransformerAttention* multiHeadAttentionLayer = { new TransformerAttention(numHeads, keySize, valueSize) };
	nn->addLayer(multiHeadAttentionLayer);
	for (int i = 0; i < numHeads; i++) {
		for (int j = 0; j < keySize; j++) {
			ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
			for (int k = 0; k < *prevSize; k++) {
				multiHeadAttentionLayer->Wq[i].r(j, k) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
		for (int j = 0; j < keySize; j++) {
			ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
			for (int k = 0; k < *prevSize; k++) {
				multiHeadAttentionLayer->Wk[i].r(j, k) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
		for (int j = 0; j < valueSize; j++) {
			ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
			for (int k = 0; k < *prevSize; k++) {
				multiHeadAttentionLayer->Wv[i].r(j, k) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
	}
	for (int i = 0; i < *prevSize - 1; i++) {
		ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < numHeads * valueSize; j++) {
			multiHeadAttentionLayer->Wo.r(i, j) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
		}
	}
}

void TransformerAttention::applyGradients(float learningRate, int t) {
	for (int i = 0; i < batchSize; i++) {
		outputOptimizer->addGradient(WoGrad[i]);
		for (int j = 0; j < numHeads; j++) {
			keyOptimizers[j]->addGradient(WkGrad[i][j]);
			queryOptimizers[j]->addGradient(WqGrad[i][j]);
			valueOptimizers[j]->addGradient(WvGrad[i][j]);
		}
	}
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

void TransformerAttention::setOptimizer(Optimizer* optimizer) {
	outputOptimizer = optimizer->clone();
	outputOptimizer->setDimensions(size, numHeads * valueSize);
	queryOptimizers = new Optimizer * [numHeads];
	keyOptimizers = new Optimizer * [numHeads];
	valueOptimizers = new Optimizer * [numHeads];
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

int TransformerAttention::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	current += size * numHeads * valueSize;
	current += numHeads * valueSize * prevSize;
	current += 2 * numHeads * keySize * prevSize;
	return current;
}