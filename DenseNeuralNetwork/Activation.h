#pragma once
#include <cmath>
class DenseLayer;

class Activation {
	public:
		static Activation* NONE;
		static Activation* SIGMOID;
		static Activation* RELU;
		static Activation* ELU;
		static Activation* SELU;
		static Activation* TANH;
		static Activation* SOFTMAX;
		static Activation* ALL_ACTIVATIONS[7];

		virtual void operate(DenseLayer* layer) = 0;
		virtual void differentiate(DenseLayer* layer) = 0;
		virtual bool isDiagonal() = 0;

};

class None : public Activation {

public:
	void operate(DenseLayer* layer);
	void differentiate(DenseLayer* layer);
	bool isDiagonal();
};

class Sigmoid : public Activation {

public:
	void operate(DenseLayer* layer);
	void differentiate(DenseLayer* layer);
	bool isDiagonal();
};

class Relu : public Activation {

public:
	void operate(DenseLayer* layer);
	void differentiate(DenseLayer* layer);
	bool isDiagonal();
};

class Elu : public Activation {

public:
	double alpha;
	Elu(double alpha);
	void operate(DenseLayer* layer);
	void differentiate(DenseLayer* layer);
	bool isDiagonal();
};

class Selu : public Activation {
public:
	void operate(DenseLayer* layer);
	void differentiate(DenseLayer* layer);
	bool isDiagonal();
};

class Tanh : public Activation {

public:
	void operate(DenseLayer* layer);
	void differentiate(DenseLayer* layer);
	bool isDiagonal();
};

class Softmax : public Activation {

public:
	void operate(DenseLayer* layer);
	void differentiate(DenseLayer* layer);
	bool isDiagonal();
};

