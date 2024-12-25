#include "Layer.h"
#include "InputLayer.h"

template<class T, class A>
bool Layer::instanceOf(A layer) {
	return dynamic_cast<T*>(layer) != NULL;
}