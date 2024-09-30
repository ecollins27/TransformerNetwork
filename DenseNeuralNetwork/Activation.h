#pragma once
#include <cmath>
#include "Optimizer.h"
class DenseLayer;

class Activation {
	public:
		const static int NUM_ACTIVATIONS = 8;
		static Activation* NONE;
		static Activation* SIGMOID;
		static Activation* RELU;
		static Activation* ELU;
		static Activation* SELU;
		static Activation* TANH;
		static Activation* SWISH;
		static Activation* SOFTMAX;
		static Activation* ALL_ACTIVATIONS[NUM_ACTIVATIONS];

		bool condenseGradient = true;
		virtual void operate(DenseLayer* layer) = 0;
		virtual void differentiate(DenseLayer* layer) = 0;
		virtual Activation* clone() = 0;
		virtual void init(DenseLayer* layer) {return;};
		virtual void setOptimizer(DenseLayer* layer, Optimizer* optimizer) { return; };
		virtual void applyGradient(DenseLayer* layer, TrainingParams* params, int t) { return; };
		virtual bool isDiagonal() { return true; };

};

class None : public Activation {

public:
	void operate(DenseLayer* layer);
	void differentiate(DenseLayer* layer);
	Activation* clone();
};

class Sigmoid : public Activation {

public:
	void operate(DenseLayer* layer);
	void differentiate(DenseLayer* layer);
	Activation* clone();
};

class Relu : public Activation {

public:
	void operate(DenseLayer* layer);
	void differentiate(DenseLayer* layer);
	Activation* clone();
};

class Elu : public Activation {

public:
	double alpha;
	Elu(double alpha);
	void operate(DenseLayer* layer);
	void differentiate(DenseLayer* layer);
	Activation* clone();
};

class Selu : public Activation {
public:
	void operate(DenseLayer* layer);
	void differentiate(DenseLayer* layer);
	Activation* clone();
};

class Tanh : public Activation {

public:
	void operate(DenseLayer* layer);
	void differentiate(DenseLayer* layer);
	Activation* clone();
};

class Swish : public Activation {

public:
	void operate(DenseLayer* layer);
	void differentiate(DenseLayer* layer);
	Activation* clone();
};

class Glu : public Activation {

public:
	double** weights;
	double** weightGradient;
	double** output;
	double** activationOutput;
	Activation* activation;
	Optimizer* optimizer;

	Glu(Activation* activation);
	void operate(DenseLayer* layer);
	void differentiate(DenseLayer* layer);
	Activation* clone();
	void init(DenseLayer* layer);
	void setOptimizer(DenseLayer* layer, Optimizer* optimizer);
	void applyGradient(DenseLayer* layer, TrainingParams* params, int t);
	bool isDiagonal();
};

class Softmax : public Activation {

public:
	void operate(DenseLayer* layer);
	void differentiate(DenseLayer* layer);
	Activation* clone();
	bool isDiagonal();
};

