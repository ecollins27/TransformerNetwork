#pragma once
#include <cmath>
#include <fstream>
#include <climits>
#include "Matrix.h"
#include "OperationQueue.h"

template<typename Type>
class ActivationOperation : public DBinary<Type, Type> {
public:

	ActivationOperation(Type& A, Type& B) : DBinary<Type, Type>(A, B) {};

};

template<typename Type>
class ActivationDifOperation : public Dnary<Type, Type> {

public:

	ActivationDifOperation(Type& input, Type& output, Type& inputGrad, Type& outputGrad) : Dnary<Type, Type>(3, 1) {
		this->in[0] = &input;
		this->in[1] = &output;
		this->in[2] = &outputGrad;
		this->out[0] = &inputGrad;
	};
};

class ActivationType {
public:
	const static int NONE = 0, SIGMOID = 1, RELU = 2, ELU = 3, SELU = 4, LOGLU = 5, TANH = 6, SWISH = 7, SOFTMAX = 8;
	static string* NAMES;
};

class Activation {

public:
	static Activation* NONE;
	static Activation* SIGMOID;
	static Activation* RELU;
	static Activation* ELU;
	static Activation* SELU;
	static Activation* LOGLU;
	static Activation* TANH;
	static Activation* SWISH;
	static Activation* SOFTMAX;

	int activationType;
	float alpha;

	Activation(int activationType, float alpha = 1);
	template<typename Type>
	Operation* getOperation(Type& input, Type& output);
	template<typename Type>
	Operation* getDifOperation(Type& input, Type& output, Type& inputGrad, Type& outputGrad);
	void save(ofstream& file);
};

template<typename Type>
class SigmoidOperation : public ActivationOperation<Type> {

public:
	SigmoidOperation(Type& A, Type& B) : ActivationOperation<Type>(A, B) { return; };
	bool operate(OperationQueue* queue, int threadID);
};

template<typename Type>
class SigmoidDifOperation : public ActivationDifOperation<Type> {

public:
	SigmoidDifOperation(Type& input, Type& output, Type& inputGrad, Type& outputGrad) : ActivationDifOperation<Type>(input, output, inputGrad, outputGrad) { return; };
	bool operate(OperationQueue* queue, int threadID);
};

template<typename Type>
class ReluOperation : public ActivationOperation<Type> {

public:
	ReluOperation(Type& A, Type& B) : ActivationOperation<Type>(A, B) { return; };
	bool operate(OperationQueue* queue, int threadID);
};

template<typename Type>
class ReluDifOperation : public ActivationDifOperation<Type> {

public:
	ReluDifOperation(Type& input, Type& output, Type& inputGrad, Type& outputGrad) : ActivationDifOperation<Type>(input, output, inputGrad, outputGrad) { return; };
	bool operate(OperationQueue* queue, int threadID);
};

template<typename Type>
class EluOperation : public ActivationOperation<Type> {

public:
	float alpha;
	EluOperation(float alpha, Type& input, Type& output) : ActivationOperation<Type>(input, output) { this->alpha = alpha; };
	bool operate(OperationQueue* queue, int threadID);
};

template<typename Type>
class EluDifOperation : public ActivationDifOperation<Type> {

public:
	float alpha;
	EluDifOperation(float alpha, Type& input, Type& output, Type& inputGrad, Type& outputGrad) : ActivationDifOperation<Type>(input, output, inputGrad, outputGrad) { this->alpha = alpha; };
	bool operate(OperationQueue* queue, int threadID);
};

template<typename Type>
class SeluOperation : public ActivationOperation<Type> {

public:
	SeluOperation(Type& A, Type& B) : ActivationOperation<Type>(A, B) { return; };
	bool operate(OperationQueue* queue, int threadID);
};

template<typename Type>
class SeluDifOperation : public ActivationDifOperation<Type> {

public:
	SeluDifOperation(Type& input, Type& output, Type& inputGrad, Type& outputGrad) : ActivationDifOperation<Type>(input, output, inputGrad, outputGrad) { return; };
	bool operate(OperationQueue* queue, int threadID);
};

template<typename Type>
class LogluOperation : public ActivationOperation<Type> {

public:
	float alpha;
	LogluOperation(float alpha, Type& input, Type& output) : ActivationOperation<Type>(input, output) { this->alpha = alpha; };
	bool operate(OperationQueue* queue, int threadID);
};

template<typename Type>
class LogluDifOperation : public ActivationDifOperation<Type> {

public:
	float alpha;
	LogluDifOperation(float alpha, Type& input, Type& output, Type& inputGrad, Type& outputGrad) : ActivationDifOperation<Type>(input, output, inputGrad, outputGrad) { this->alpha = alpha; };
	bool operate(OperationQueue* queue, int threadID);
};

template<typename Type>
class TanhOperation : public ActivationOperation<Type> {

public:
	TanhOperation(Type& A, Type& B) : ActivationOperation<Type>(A, B) { return; };
	bool operate(OperationQueue* queue, int threadID);
};

template<typename Type>
class TanhDifOperation : public ActivationDifOperation<Type> {

public:
	TanhDifOperation(Type& input, Type& output, Type& inputGrad, Type& outputGrad) : ActivationDifOperation<Type>(input, output, inputGrad, outputGrad) { return; };
	bool operate(OperationQueue* queue, int threadID);
};

template<typename Type>
class SwishOperation : public ActivationOperation<Type> {

public:
	float alpha;
	SwishOperation(float alpha, Type& input, Type& output) : ActivationOperation<Type>(input, output) { this->alpha = alpha; };
	bool operate(OperationQueue* queue, int threadID);
};

template<typename Type>
class SwishDifOperation : public ActivationDifOperation<Type> {

public:
	float alpha;
	SwishDifOperation(float alpha, Type& input, Type& output, Type& inputGrad, Type& outputGrad) : ActivationDifOperation<Type>(input, output, inputGrad, outputGrad) { this->alpha = alpha; };
	bool operate(OperationQueue* queue, int threadID);
};

template<typename Type>
class SoftmaxOperation : public ActivationOperation<Type> {

public:
	SoftmaxOperation(Type& A, Type& B) : ActivationOperation<Type>(A, B) { return; };
	bool operate(OperationQueue* queue, int threadID);
};

template<typename Type>
class SoftmaxDifOperation : public ActivationDifOperation<Type> {

public:
	SoftmaxDifOperation(Type& input, Type& output, Type& inputGrad, Type& outputGrad) : ActivationDifOperation<Type>(input, output, inputGrad, outputGrad) { return; };
	bool operate(OperationQueue* queue, int threadID);
};

#include "Activation.inl"