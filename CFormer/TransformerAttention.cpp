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
	MatrixBatch::multiplyABC(prevLayer->neurons[num], Wq, Q[num], true);
	MatrixBatch::multiplyAtBC(prevLayer->neurons[num], Wk, K[num], true);
	MatrixBatch::multiplyABtC(Q[num], K[num], A[num], true);
	A[num].scale(scalar);
	softmax->operate(A[num], Ao[num]);
	MatrixBatch::multiplyAtBC(prevLayer->neurons[num], Wv, V[num], true);
	MatrixBatch::multiplyABC(Ao[num], V[num], AcSub[num], true);
	Matrix2::multiplyABC(Ac[num], Wo, neurons[num], true);



	//MatrixBatch::multiplyABtC(prevLayer->neurons[num], Wq, Q[num], true);
	//MatrixBatch::multiplyABtC(prevLayer->neurons[num], Wk, K[num], true);
	//MatrixBatch::multiplyABtC(Q[num], K[num], A[num], true);
	//A[num].scale(scalar);
	//softmax->operate(A[num], A[num]);
	//MatrixBatch::multiplyABtC(Wv, prevLayer->neurons[num], V[num], true);
	//MatrixBatch::multiplyABtC(A[num], V[num], AcSub[num], true);
	//Matrix2::multiplyABtC(Ac[num], Wo, neurons[num], true);
}

void TransformerAttention::backPropagate(int num) {
	float scalar = 1.0 / sqrt(keySize);
	Matrix2::multiplyABtC(Wo, neuronGradient[num], AcGrad[num], true);
	Matrix2::multiplyABC(Ac[num], neuronGradient[num], WoGrad[num], true);

	MatrixBatch::multiplyAtBtC(A[num], AcSubGrad[num], VGrad[num], true);
	MatrixBatch::multiplyAtBtC(AcSubGrad[num], V[num], AoGrad[num], true);
	MatrixBatch::multiplyABtC(VGrad[num], Wv, prevNeuronGradient[num], true);
	MatrixBatch::multiplyAtBC(prevLayer->neurons[num], VGrad[num], WvGrad[num], true);
	softmax->differentiate(A[num], Ao[num], AGrad[num], AoGrad[num]);
	MatrixBatch::multiplyABC(AGrad[num], K[num], QGrad[num], true);
	MatrixBatch::multiplyAtBC(AGrad[num], Q[num], KGrad[num], true);
	QGrad[num].scale(scalar);
	KGrad[num].scale(scalar);
	MatrixBatch::multiplyABtC(KGrad[num], Wk, prevNeuronGradient[num], false);
	MatrixBatch::multiplyAtBC(prevLayer->neurons[num], KGrad[num], WkGrad[num], true);
	MatrixBatch::multiplyABtC(QGrad[num], Wq, prevNeuronGradient[num], false);
	MatrixBatch::multiplyAtBC(prevLayer->neurons[num], QGrad[num], WqGrad[num], true);
	prevNeuronGradient[num].condense(prevLayer->neuronGradient[num]);
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
	NormalFill normal = NormalFill(0, std);
	Wq = MatrixBatch(normal, numHeads, keySize, prevSize);
	Wq.deallocateHost();
	Wk = MatrixBatch(normal, numHeads, keySize, prevSize);
	Wk.deallocateHost();
	Wv = MatrixBatch(normal, numHeads, valueSize, prevSize);
	Wv.deallocateHost();
	Wo = Matrix2(normal, numHeads * valueSize, size);
	Wo.deallocateHost();
}

void TransformerAttention::setBatchSize(int batchSize) {
	Layer2D::initNeurons(batchSize);
	for (int i = 0; i < batchSize; i++) {
		prevLayer->neurons[i].allocateBatchDevice(batchSize);
	}
	WqGrad = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, prevSize, keySize, false);
	WkGrad = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, prevSize, keySize, false);
	WvGrad = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, prevSize, valueSize, false);
	WoGrad = Matrix2::allocateMatrixArray(batchSize, numHeads * valueSize, size, false);

	outputOptimizer->setBatchSize(batchSize, WoGrad);
	queryOptimizers->setBatchSize(batchSize, WqGrad);
	keyOptimizers->setBatchSize(batchSize, WkGrad);
	valueOptimizers->setBatchSize(batchSize, WvGrad);

	K = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, keySize, false);
	KGrad = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, keySize, false);
	Q = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, keySize, false);
	QGrad = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, keySize, false);
	V = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, valueSize, false);
	VGrad = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, valueSize, false);
	A = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, maxNumTokens, false);
	AGrad = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, maxNumTokens, false);
	Ao = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, maxNumTokens, false);
	AoGrad = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, maxNumTokens, false);
	Ac = Matrix2::allocateMatrixArray(batchSize, numHeads * valueSize, maxNumTokens, false);
	AcGrad = Matrix2::allocateMatrixArray(batchSize, numHeads * valueSize, maxNumTokens, false);
	AcSub = new MatrixBatch[batchSize];
	AcSubGrad = new MatrixBatch[batchSize];
	for (int i = 0; i < batchSize; i++) {
		AcSub[i] = Ac[i].subMatrixBatch(numHeads, valueSize);
		AcSubGrad[i] = AcGrad[i].subMatrixBatch(numHeads, valueSize);
	}
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void TransformerAttention::save(ofstream& file) {
	file << LAYER_NAME << ",";
	file << numHeads << "," << keySize << "," << valueSize << ",\n";
	Wq.allocateHost();
	Wk.allocateHost();
	Wv.allocateHost();
	for (int i = 0; i < numHeads; i++) {
		for (int j = 0; j < prevSize; j++) {
			for (int k = 0; k < keySize; k++) {
				file << Wq(i, j, k) << ",";
			}
			file << "\n";
		}
		for (int j = 0; j < prevSize; j++) {
			for (int k = 0; k < keySize; k++) {
				file << Wk(i, j, k) << ",";
			}
			file << "\n";
		}
		for (int j = 0; j < prevSize; j++) {
			for (int k = 0; k < valueSize; k++) {
				file << Wv(i, j, k) << ",";
			}
			file << "\n";
		}
	}
	Wq.deallocateHost();
	Wk.deallocateHost();
	Wv.deallocateHost();
	Wo.allocateHost();
	for (int i = 0; i < numHeads * valueSize; i++) {
		for (int j = 0; j < size; j++) {
			file << Wo(i, j) << ",";
		}
		file << "\n";
	}
	Wo.deallocateHost();
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
	multiHeadAttentionLayer->Wq.allocateHost();
	multiHeadAttentionLayer->Wk.allocateHost();
	multiHeadAttentionLayer->Wv.allocateHost();
	for (int i = 0; i < numHeads; i++) {
		for (int j = 0; j < *prevSize; j++) {
			ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
			for (int k = 0; k < keySize; k++) {
				multiHeadAttentionLayer->Wq(i, j, k) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
		for (int j = 0; j < *prevSize; j++) {
			ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
			for (int k = 0; k < keySize; k++) {
				multiHeadAttentionLayer->Wk(i, j, k) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
		for (int j = 0; j < *prevSize; j++) {
			ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
			for (int k = 0; k < valueSize; k++) {
				multiHeadAttentionLayer->Wv(i, j, k) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
	}
	multiHeadAttentionLayer->Wq.deallocateHost();
	multiHeadAttentionLayer->Wk.deallocateHost();
	multiHeadAttentionLayer->Wv.deallocateHost();
	multiHeadAttentionLayer->Wo.allocateHost();
	for (int i = 0; i < numHeads * valueSize; i++) {
		ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize - 1; j++) {
			multiHeadAttentionLayer->Wo(i, j) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
		}
	}
	multiHeadAttentionLayer->Wo.deallocateHost();
}

void TransformerAttention::setNumTokens(int* numTokens) {
	this->numTokens = numTokens;
	updateNeuronDimensions();
	for (int i = 0; i < batchSize; i++) {
		prevNeuronGradient[i].setHeight(numTokens[i]);
		prevNeuronGradient[i].constantFill(0);
		K[i].setHeight(numTokens[i]);
		KGrad[i].setHeight(numTokens[i]);
		Q[i].setHeight(numTokens[i]);
		QGrad[i].setHeight(numTokens[i]);
		V[i].setWidth(numTokens[i]);
		VGrad[i].setWidth(numTokens[i]);
		A[i].setDims(numTokens[i], numTokens[i]);
		AGrad[i].setDims(numTokens[i], numTokens[i]);
		Ao[i].setDims(numTokens[i], numTokens[i]);
		AoGrad[i].setDims(numTokens[i], numTokens[i]);
		Ac[i].setWidth(numTokens[i]);
		AcGrad[i].setWidth(numTokens[i]);
		AcSub[i].setWidth(numTokens[i]);
		AcSubGrad[i].setWidth(numTokens[i]);
	}
	if (nextLayer != NULL && instanceOf<Layer2D>(nextLayer)) {
		((Layer2D*)nextLayer)->setNumTokens(numTokens);
	}
}

void TransformerAttention::applyGradients(float learningRate, int t) {
	outputOptimizer->condenseGradients();
	keyOptimizers->condenseGradients();
	queryOptimizers->condenseGradients();
	valueOptimizers->condenseGradients();
	outputOptimizer->applyGradient(Wo, t, learningRate);
	queryOptimizers->applyGradient(Wq, t, learningRate);
	keyOptimizers->applyGradient(Wk, t, learningRate);
	valueOptimizers->applyGradient(Wv, t, learningRate);
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void TransformerAttention::setOptimizer(Optimizer* optimizer) {
	outputOptimizer = optimizer->clone();
	outputOptimizer->setDimensions(numHeads * valueSize, size);
	queryOptimizers = optimizer->cloneBatch();
	queryOptimizers->setDimensions(numHeads, prevSize, keySize);
	keyOptimizers = optimizer->cloneBatch();
	keyOptimizers->setDimensions(numHeads, prevSize, keySize);
	valueOptimizers = optimizer->cloneBatch();
	valueOptimizers->setDimensions(numHeads, prevSize, valueSize);
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