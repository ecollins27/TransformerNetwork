#include "FillFunction.h"

ConstantFillFunction FillFunction::ZERO_FILL = ConstantFillFunction(0);
NormalFillFunction FillFunction::UNIT_NORMAL_FILL = NormalFillFunction(0, 1);
UniformFillFunction FillFunction::UNIT_UNIFORM_FILL = UniformFillFunction(0, 1);

ConstantFillFunction::ConstantFillFunction(float value) {
	this->value = value;
}

float ConstantFillFunction::operator()(int i, int j) {
	return value;
}

NormalFillFunction::NormalFillFunction(float mean, float stdDeviation) {
	distribution = new normal_distribution<float>(mean, stdDeviation);
}

float NormalFillFunction::operator()(int i, int j) {
	return (*distribution)(generator);
}

UniformFillFunction::UniformFillFunction(float lowerBound, float upperBound) {
	distribution = new uniform_real_distribution<float>(lowerBound, upperBound);
}

float UniformFillFunction::operator()(int i, int j) {
	return (*distribution)(generator);
}