#include "Layer.h"
#include "InputLayer.h"

void Layer::setTrainable(bool trainable) {
	this->trainable = trainable;
	if (nextLayer != NULL) {
		nextLayer->setTrainable(trainable);
	}
}

int Layer::getLayerIndex() {
	Layer* layer = this;
	int index = 0;
	while (dynamic_cast<InputLayer*>(layer) == NULL) {
		index++;
		layer = layer->prevLayer;
	}
	return index;
}