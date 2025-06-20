#include "FillFunction.h"

ConstantFill FillFunction::ZERO_FILL = ConstantFill(0);
NormalFill FillFunction::UNIT_NORMAL_FILL = NormalFill(0, 1);
UniformFill FillFunction::UNIT_UNIFORM_FILL = UniformFill(0, 1);

ConstantFill::ConstantFill(float value) {
	this->value = value;
}

float ConstantFill::operator()(int i, int j) {
	return value;
}

NormalFill::NormalFill(float mean, float stdDeviation) {
	distribution = new normal_distribution<float>(mean, stdDeviation);
}

float NormalFill::operator()(int i, int j) {
	return (*distribution)(generator);
}

UniformFill::UniformFill(float lowerBound, float upperBound) {
	distribution = new uniform_real_distribution<float>(lowerBound, upperBound);
}

float UniformFill::operator()(int i, int j) {
	return (*distribution)(generator);
}