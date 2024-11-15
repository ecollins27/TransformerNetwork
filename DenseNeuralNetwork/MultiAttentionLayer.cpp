#include "MultiAttentionLayer.h"

MultiAttentionLayer::MultiAttentionLayer(int numHeads, int keySize, int valueSize, int maxTokenSize) {
	this->numHeads = numHeads;
	this->keySize = keySize;
	this->valueSize = valueSize;
	this->maxTokenSize = maxTokenSize;
}

void MultiAttentionLayer::predict() {
	double scalar = 1.0 / sqrt(keySize);
	for (int i = 0; i < numHeads; i++) {
		Matrix::matrixMultiplyABtC(batchSize, size, keySize, prevLayer->neurons, queryWeights[i], queries[i], true);
		Matrix::matrixMultiplyABtC(batchSize, size, keySize, prevLayer->neurons, keyWeights[i], keys[i], true);
		Matrix::matrixMultiplyABtC(batchSize, keySize, batchSize, queries[i], keys[i], activations[i], true);
		Matrix::scale(batchSize, batchSize, activations[i], scalar);
		softmax->operate(batchSize, batchSize, activations[i], activationOutputs[i]);
		Matrix::matrixMultiplyABtC(valueSize, size, batchSize, valueWeights[i], prevLayer->neuronsTranspose, values[i], true);
		Matrix::matrixMultiplyABtC(valueSize, batchSize, batchSize, values[i], activationOutputs[i], &attentionConcatTranspose[i * valueSize], true);
	}
	Matrix::transpose(numHeads * valueSize, batchSize, attentionConcatTranspose, attentionConcat);
	Matrix::matrixMultiplyABtC(batchSize, numHeads * valueSize, size, attentionConcat, outputWeights, neurons, true);
	Matrix::transpose(batchSize, size, neurons, neuronsTranspose);
	if (nextLayer != NULL) {
		nextLayer->predict();
	}
}

void MultiAttentionLayer::forwardPropagate() {
	double scalar = 1.0 / sqrt(keySize);
	for (int i = 0; i < numHeads; i++) {
		Matrix::matrixMultiplyABtC(batchSize, size, keySize, prevLayer->neurons, queryWeights[i], queries[i], true);
		Matrix::matrixMultiplyABtC(batchSize, size, keySize, prevLayer->neurons, keyWeights[i], keys[i], true);
		Matrix::matrixMultiplyABtC(batchSize, keySize, batchSize, queries[i], keys[i], activations[i], true);
		Matrix::scale(batchSize, batchSize, activations[i], scalar);
		softmax->operate(batchSize, batchSize, activations[i], activationOutputs[i]);
		Matrix::matrixMultiplyABtC(valueSize, size, batchSize, valueWeights[i], prevLayer->neuronsTranspose, values[i], true);
		Matrix::matrixMultiplyABtC(valueSize, batchSize, batchSize, values[i], activationOutputs[i], &attentionConcatTranspose[i * valueSize], true);
	}
	Matrix::transpose(numHeads * valueSize, batchSize, attentionConcatTranspose, attentionConcat);
	Matrix::matrixMultiplyABtC(batchSize, numHeads * valueSize, size, attentionConcat, outputWeights, neurons, true);
	Matrix::transpose(batchSize, size, neurons, neuronsTranspose);
	if (nextLayer != NULL) {
		nextLayer->forwardPropagate();
	}
}

void MultiAttentionLayer::backPropagate() {

	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
}

void MultiAttentionLayer::setPrevLayer(Layer* prevLayer) {
	this->prevLayer = prevLayer;
	this->prevSize = prevLayer->size;
	this->size = prevSize;
	float std = 1.0 / maxTokenSize;
	Matrix::FillFunction* fillFunction = { new Matrix::NormalFill(0, std) };
	queryWeights = Matrix::allocate3DMatrix(fillFunction, numHeads, keySize, size);
	queryWeightGradients = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, keySize, size);
	keyWeights = Matrix::allocate3DMatrix(fillFunction, numHeads, keySize, size);
	keyWeightGradients = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, keySize, size);
	valueWeights = Matrix::allocate3DMatrix(fillFunction, numHeads, valueSize, size);
	valueWeightGradients = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, valueSize, size);
	outputWeights = Matrix::allocateMatrix(fillFunction, size, numHeads * valueSize);
	outputWeightGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, numHeads * valueSize);

	queries = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxTokenSize, keySize);
	keys = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxTokenSize, keySize);
	values = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, valueSize, maxTokenSize);
	activations = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxTokenSize, maxTokenSize);
	activationOutputs = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, numHeads, maxTokenSize, maxTokenSize);
	attentionConcat = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxTokenSize, numHeads * valueSize);
	attentionConcatTranspose = Matrix::allocateMatrix(Matrix::ZERO_FILL, numHeads * valueSize, maxTokenSize);
	activationGradients = Matrix::allocate4DMatrix(Matrix::ZERO_FILL, numHeads, maxTokenSize, maxTokenSize, maxTokenSize);
	neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxTokenSize, size);
	neuronsTranspose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, maxTokenSize);
	neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxTokenSize, size);
}
void MultiAttentionLayer::setNextLayer(Layer* nextLayer) {
	this->nextLayer = nextLayer;
}

void MultiAttentionLayer::setBatchSize(int batchSize) {
	this->batchSize = batchSize;
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void MultiAttentionLayer::applyGradients(float learningRate, int t) {
	optimizer->applyGradient(outputWeightGradient, outputWeights, t, learningRate);
	for (int i = 0; i < numHeads; i++) {
		queryOptimizers[i]->applyGradient(queryWeightGradients[i], queryWeights[i], t, learningRate);
		keyOptimizers[i]->applyGradient(keyWeightGradients[i], keyWeights[i], t, learningRate);
		valueOptimizers[i]->applyGradient(valueWeightGradients[i], valueWeights[i], t, learningRate);
	}
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void MultiAttentionLayer::setOptimizer(Optimizer* optimizer) {
	this->optimizer = optimizer->clone();
	this->optimizer->setDimensions(size, numHeads * valueSize);
	queryOptimizers = (Optimizer**)malloc(numHeads * sizeof(Optimizer*));
	keyOptimizers = (Optimizer**)malloc(numHeads * sizeof(Optimizer*));
	valueOptimizers = (Optimizer**)malloc(numHeads * sizeof(Optimizer*));
	for (int i = 0; i < numHeads; i++) {
		queryOptimizers[i] = optimizer->clone();
		queryOptimizers[i]->setDimensions(keySize, size);
		keyOptimizers[i] = optimizer->clone();
		keyOptimizers[i]->setDimensions(keySize, size);
		valueOptimizers[i] = optimizer->clone();
		valueOptimizers[i]->setDimensions(valueSize, size);
	}
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

void MultiAttentionLayer::save(ofstream& file) {
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

int MultiAttentionLayer::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	return current + 2 * numHeads * valueSize * size + 2 * numHeads * keySize * size;
}