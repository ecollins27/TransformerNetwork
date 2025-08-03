#pragma once
#include <iostream>
#include <random>

using namespace std;

class ConstantFillFunction;
class NormalFillFunction;
class UniformFillFunction;

class FillFunction {

public:

	static ConstantFillFunction ZERO_FILL;
	static NormalFillFunction UNIT_NORMAL_FILL;
	static UniformFillFunction UNIT_UNIFORM_FILL;

	virtual float operator()(int i, int j) {
		return 0.1;
	};
};


class ConstantFillFunction : public FillFunction {
public:
	float value;
	ConstantFillFunction(float value);
	float operator()(int i, int j) override;
};

class NormalFillFunction : public FillFunction {
public:
	default_random_engine generator;
	normal_distribution<float>* distribution;

	NormalFillFunction(float mean, float stdDeviation);
	float operator()(int i, int j) override;
};

class UniformFillFunction : public FillFunction {
public:
	default_random_engine generator;
	uniform_real_distribution<float>* distribution;

	UniformFillFunction(float lowerBound, float upperBound);
	float operator()(int i, int j) override;
};