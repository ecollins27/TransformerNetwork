#include "Layer1D.h"

Layer1D::~Layer1D() {
	neurons.free();
	neuronGradient.free();
	Layer::~Layer();
}