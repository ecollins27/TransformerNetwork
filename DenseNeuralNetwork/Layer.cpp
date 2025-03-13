#include "Layer.h"

Layer::~Layer() {
	if (nextLayer != NULL) {
		delete nextLayer;
	}
}