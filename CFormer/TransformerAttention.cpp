#include "TransformerAttention.h"
#include "Model.h"
#include "ModelParser.h"

const string TransformerAttention::LAYER_NAME = "TransformerAttention";

TransformerAttention::TransformerAttention(int numHeads, int keySize, int valueSize) {
	this->numHeads = numHeads;
	this->keySize = keySize;
	this->valueSize = valueSize;
	softmax = Activation::SOFTMAX;
}

void TransformerAttention::initPropagationQueue(OperationQueue& queue) {
	float scalar = 1.0 / sqrt(keySize);
	for (int i = 0; i < batchSize; i++) {
		queue.enqueue(new MultiplyABC(prevLayer->neurons[i], Wq, Q[i], true));
		queue.enqueue(new MultiplyABC(prevLayer->neurons[i], Wk, K[i], true));
		queue.enqueue(new MultiplyABtC(Q[i], K[i], A[i], true));
		queue.enqueue(softmax->getOperation(A[i], Ao[i])); 
		queue.enqueue(new MultiplyABC(prevLayer->neurons[i], Wv, V[i], true));
		queue.enqueue(new MultiplyABC(Ao[i], V[i], AcSub[i], true));
		queue.enqueue(new MultiplyABC(Ac[i], Wo, neurons[i], true));
	}
}

void TransformerAttention::initBackPropQueue(OperationQueue& queue) {

}

//void TransformerAttention::backPropagate(int num) {
//	float scalar = 1.0 / sqrt(keySize);
//	Matrix::multiplyABtC(Wo, neuronGradient[num], AcGrad[num], true);
//	Matrix::multiplyABC(Ac[num], neuronGradient[num], WoGrad[num], true);
//
//	MatrixBatch::multiplyAtBtC(A[num], AcSubGrad[num], VGrad[num], true);
//	MatrixBatch::multiplyAtBtC(AcSubGrad[num], V[num], AoGrad[num], true);
//	MatrixBatch::multiplyABtC(VGrad[num], Wv, prevNeuronGradient[num], true);
//	MatrixBatch::multiplyAtBC(prevLayer->neurons[num], VGrad[num], WvGrad[num], true);
//	softmax->differentiate(A[num], Ao[num], AGrad[num], AoGrad[num]);
//	MatrixBatch::multiplyABC(AGrad[num], K[num], QGrad[num], true);
//	MatrixBatch::multiplyAtBC(AGrad[num], Q[num], KGrad[num], true);
//	QGrad[num].scale(scalar);
//	KGrad[num].scale(scalar);
//	MatrixBatch::multiplyABtC(KGrad[num], Wk, prevNeuronGradient[num], false);
//	MatrixBatch::multiplyAtBC(prevLayer->neurons[num], KGrad[num], WkGrad[num], true);
//	MatrixBatch::multiplyABtC(QGrad[num], Wq, prevNeuronGradient[num], false);
//	MatrixBatch::multiplyAtBC(prevLayer->neurons[num], QGrad[num], WqGrad[num], true);
//	prevNeuronGradient[num].condense(prevLayer->neuronGradient[num]);
//	prevLayer->backPropagate(num);
//}

void TransformerAttention::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
	size = prevLayer->size;
	prevSize = prevLayer->size + 1;
	float std = 1.0 / size;
	NormalFillFunction normal = NormalFillFunction(0, std);
	Wq = MatrixBatch(normal, numHeads, prevSize, keySize);
	Wk = MatrixBatch(normal, numHeads, prevSize, keySize);
	Wv = MatrixBatch(normal, numHeads, prevSize, valueSize);
	Wo = Matrix(normal, numHeads * valueSize, size);
}

void TransformerAttention::setBatchSize(int batchSize) {
	Layer2D::initNeurons(batchSize);
	WqGrad = MatrixBatch(numHeads, prevSize, keySize);
	WkGrad = MatrixBatch(numHeads, prevSize, keySize);
	WvGrad = MatrixBatch(numHeads, prevSize, valueSize);
	WoGrad = Matrix(numHeads * valueSize, size);

	K = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, keySize);
	KGrad = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, keySize);
	Q = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, keySize);
	QGrad = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, keySize);
	V = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, valueSize);
	VGrad = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, valueSize);
	A = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, maxNumTokens);
	AGrad = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, maxNumTokens);
	Ao = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, maxNumTokens);
	AoGrad = MatrixBatch::allocateMatrixBatchArray(batchSize, numHeads, maxNumTokens, maxNumTokens);
	Ac = Matrix::allocateMatrixArray(batchSize, maxNumTokens, numHeads * valueSize);
	AcGrad = Matrix::allocateMatrixArray(batchSize, maxNumTokens, numHeads * valueSize);
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
	for (int i = 0; i < numHeads * valueSize; i++) {
		for (int j = 0; j < size; j++) {
			file << Wo(i, j) << ",";
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
		for (int j = 0; j < *prevSize; j++) {
			ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
			for (int k = 0; k < keySize; k++) {
				multiHeadAttentionLayer->Wq(i, j, k) = ModelParser::getNextfloat(line, commaIndex, newCommaIndex);
			}
		}
		for (int j = 0; j < *prevSize; j++) {
			ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
			for (int k = 0; k < keySize; k++) {
				multiHeadAttentionLayer->Wk(i, j, k) = ModelParser::getNextfloat(line, commaIndex, newCommaIndex);
			}
		}
		for (int j = 0; j < *prevSize; j++) {
			ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
			for (int k = 0; k < valueSize; k++) {
				multiHeadAttentionLayer->Wv(i, j, k) = ModelParser::getNextfloat(line, commaIndex, newCommaIndex);
			}
		}
	}
	for (int i = 0; i < numHeads * valueSize; i++) {
		ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize - 1; j++) {
			multiHeadAttentionLayer->Wo(i, j) = ModelParser::getNextfloat(line, commaIndex, newCommaIndex);
		}
	}
}

void TransformerAttention::setNumTokens(int* numTokens) {
	this->numTokens = numTokens;
	updateNeuronDimensions();
	for (int i = 0; i < batchSize; i++) {
		prevNeuronGradient[i].setHeight(numTokens[i]);
		prevNeuronGradient[i].fill(FillFunction::ZERO_FILL);
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

void TransformerAttention::initApplicationQueue(OperationQueue& queue, float learningRate, int& t) {
	outputOptimizer->initApplicationQueue(queue, Wo, learningRate, batchSize, t);
	queryOptimizers->initApplicationQueue(queue, Wq, learningRate, batchSize, t);
	keyOptimizers->initApplicationQueue(queue, Wk, learningRate, batchSize, t);
	valueOptimizers->initApplicationQueue(queue, Wv, learningRate, batchSize, t);
	if (nextLayer != NULL) {
		nextLayer->initApplicationQueue(queue, learningRate, t);
	}
}

void TransformerAttention::setOptimizer(Optimizer<>* optimizer) {
	outputOptimizer = optimizer->clone<Matrix>();
	outputOptimizer->setDimensions(1, numHeads * valueSize, size);
	queryOptimizers = optimizer->clone<MatrixBatch>();
	queryOptimizers->setDimensions(numHeads, prevSize, keySize);
	keyOptimizers = optimizer->clone<MatrixBatch>();
	keyOptimizers->setDimensions(numHeads, prevSize, keySize);
	valueOptimizers = optimizer->clone<MatrixBatch>();
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