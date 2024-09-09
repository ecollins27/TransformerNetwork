#include "Layer.h"

void Layer::setTrainable(bool trainable) {
	this->trainable = trainable;
	if (nextLayer != NULL) {
		nextLayer->setTrainable(trainable);
	}
}