#include "ResidualSave2D.h"
#include "Model.h"

const string ResidualSave2D::LAYER_NAME = "ResidualSave2D";

void ResidualSave2D::initPropagationQueue(OperationQueue& queue) {
	for (int i = 0; i < batchSize; i++) {
		queue.enqueue(new CopyTo(prevLayer->neurons[i], neurons[i]));
	}
}

void ResidualSave2D::initBackPropQueue(OperationQueue& queue) {
	return;
}

void ResidualSave2D::initBackPropQueueWithResidual(OperationQueue& queue) {
	for (int i = 0; i < batchSize; i++) {
		queue.enqueue(new CopyTo(neuronGradient[i], prevLayer->neuronGradient[i]));
	}
	prevLayer->initBackPropQueue(queue);
}

void ResidualSave2D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
	prevSize = prevLayer->size + 1;
	size = prevLayer->size;
}

void ResidualSave2D::save(ofstream& file) {
	file << LAYER_NAME << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void ResidualSave2D::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	ResidualSave2D* residualSave = { new ResidualSave2D() };
	nn->addLayer(residualSave);
}