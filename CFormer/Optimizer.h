#pragma once
#include "TrainingParams.h"
#include "Matrix.h"
#include "OperationQueue.h"

template<typename Type>
class Optimizer {

public:
	static Optimizer<>* GRADIENT_DESCENT;
	static Optimizer<>* MOMENTUM;
	static Optimizer<>* ADAM;
	static Optimizer<>* ADEMAMIX;

	int height, width, batchSize;
	Type weightGradient;
	float regConstant;

	virtual void initApplicationQueue(OperationQueue& queue, Type& weights, float learningRate, int batchSize, int& t) = 0;
	template<typename Type2>
	Optimizer<Type2>* clone() { return NULL; };
	virtual void setDimensions(int batchSize, int height, int width) = 0;
};

template<typename Type = Matrix>
class GradientDescent : public Optimizer<Type> {

public:
	GradientDescent(float weightDecay);
	void initApplicationQueue(OperationQueue& queue, Type& weights, float learningRate, int batchSize, int& t);
	template<typename Type2>
	Optimizer<Type2>* clone() {
		return new GradientDescent<Type2>(this->regConstant);
	}
	void setDimensions(int batchSize, int height, int width);
};

template<typename Type = Matrix>
class Momentum : public Optimizer<Type> {

public:
	float beta;
	Type M;

	Momentum(float beta, float weightDecay);
	void initApplicationQueue(OperationQueue& queue, Type& weights, float learningRate, int batchSize, int& t);
	template<typename Type2>
	Optimizer<Type2>* clone() {
		return new Momentum<Type2>(this->beta, this->regConstant);
	}
	void setDimensions(int batchSize, int height, int width);
};

template<typename Type = Matrix>
class Adam : public Optimizer<Type> {

public:
	float beta1, beta2;
	Type M;
	Type S;

	Adam(float beta1, float beta2, float weightDecay);
	void initApplicationQueue(OperationQueue& queue, Type& weights, float learningRate, int batchSize, int& t);
	template<typename Type2>
	Optimizer<Type2>* clone() {
		return new Adam<Type2>(this->beta1, this->beta2, this->regConstant);
	}
	void setDimensions(int batchSize, int height, int width);
};

template<typename Type = Matrix>
class AdEMAMix : public Optimizer<Type> {

public:

	float beta1, beta2, beta3, alpha;
	Type M1;
	Type M2;
	Type S;

	AdEMAMix(float beta1, float beta2, float beta3, float alpha, float weightDecay);
	void initApplicationQueue(OperationQueue& queue, Type& weights, float learningRate, int batchSize, int& t);
	template<typename Type2>
	Optimizer<Type2>* clone() {
		return new AdEMAMix<Type2>(this->beta1, this->beta2, this->beta3, this->regConstant);
	}
	void setDimensions(int batchSize, int height, int width);
};

template<typename Type>

class LinearCombo2 : public DTrinary<Type, Type, Type> {

public:
	float c1, c2;
	LinearCombo2(float c1, Type& A, float c2, Type& B, Type& C) : DTrinary<Type, Type, Type>(A, B, C) { this->c1 - c1; this->c2 = c2; };
	bool operate(OperationQueue* queue, int threadID);
};

template<typename Type>
class AdamOperation : public Dnary<Type, Type> {

public:
	float learningRate, beta1, beta2;
	int* t;
	AdamOperation(Type& weights, Type& M, Type& S, float beta1, float beta2, float learningRate, int& t) : Dnary<Type, Type>(3, 1) {
		this->in[0] = &weights;
		this->in[1] = &M;
		this->in[2] = &S;
		this->out[0] = &weights;
		this->beta1 = beta1;
		this->beta2 = beta2;
		this->learningRate = learningRate;
		this->t = &t;
	}
	bool operate(OperationQueue* queue, int threadID);
};

template<typename Type>
class AdEMAMixOperation : public Dnary<Type, Type> {

public:
	float alpha, learningRate, beta1, beta2, beta3;
	int* t;
	AdEMAMixOperation(Type& weights, Type& M1, Type& M2, Type& S, float beta1, float beta2, float beta3, float learningRate, float alpha, int& t) : Dnary<Type, Type>(4, 1) {
		this->in[0] = &weights;
		this->in[1] = &M1;
		this->in[2] = &M2;
		this->in[3] = &S;
		this->out[0] = &weights;
		this->beta1 = beta1;
		this->beta2 = beta2;
		this->beta3 = beta3;
		this->alpha = alpha;
		this->learningRate = learningRate;
		this->t = &t;
	}
	bool operate(OperationQueue* queue, int threadID);
};