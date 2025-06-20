#pragma once
#include <iostream>
#include <random>

using namespace std;

class ConstantFill;
class NormalFill;
class UniformFill;

class FillFunction {

public:

	static ConstantFill ZERO_FILL;
	static NormalFill UNIT_NORMAL_FILL;
	static UniformFill UNIT_UNIFORM_FILL;

	virtual float operator()(int i, int j) {
		return 0;
	};
};


class ConstantFill : public FillFunction {
public:
	float value;
	ConstantFill(float value);
	float operator()(int i, int j) override;
};

class NormalFill : public FillFunction {
public:
	default_random_engine generator;
	normal_distribution<float>* distribution;

	NormalFill(float mean, float stdDeviation);
	float operator()(int i, int j) override;
};

class UniformFill : public FillFunction {
public:
	default_random_engine generator;
	uniform_real_distribution<float>* distribution;

	UniformFill(float lowerBound, float upperBound);
	float operator()(int i, int j) override;
};