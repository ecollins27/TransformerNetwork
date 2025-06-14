#pragma once
#include <cmath>
#include <fstream>
#include <climits>
#include "Matrix3D.h"

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
		virtual void operate(int batchSize, int size, Matrix input, Matrix output) = 0;
		virtual void differentiate(int batchSize, int size, Matrix A, Matrix Ao, Matrix& AGrad, Matrix AoGrad) = 0;
		virtual Activation* clone() = 0;
		virtual bool isDiagonal() { return true; };
		virtual void save(ofstream& file) {
			string name(& typeid(*this).name()[6]);
			file << name << ",";
		};

};

class None : public Activation {

public:
	void operate(int batchSize, int size, Matrix input, Matrix output);
	void differentiate(int batchSize, int size, Matrix A, Matrix Ao, Matrix& AGrad, Matrix AoGrad);
	Activation* clone();
};

class Sigmoid : public Activation {

public:
	void operate(int batchSize, int size, Matrix input, Matrix output);
	void differentiate(int batchSize, int size, Matrix A, Matrix Ao, Matrix& AGrad, Matrix AoGrad);
	Activation* clone();
};

class Relu : public Activation {

public:
	void operate(int batchSize, int size, Matrix input, Matrix output);
	void differentiate(int batchSize, int size, Matrix A, Matrix Ao, Matrix& AGrad, Matrix AoGrad);
	Activation* clone();
};

class Elu : public Activation {

public:
	float alpha;
	Elu(float alpha);
	void operate(int batchSize, int size, Matrix input, Matrix output);
	void differentiate(int batchSize, int size, Matrix A, Matrix Ao, Matrix& AGrad, Matrix AoGrad);
	Activation* clone();
	void save(ofstream& file);
};

class Selu : public Activation {
public:
	void operate(int batchSize, int size, Matrix input, Matrix output);
	void differentiate(int batchSize, int size, Matrix A, Matrix Ao, Matrix& AGrad, Matrix AoGrad);
	Activation* clone();
};

class Tanh : public Activation {

public:
	void operate(int batchSize, int size, Matrix input, Matrix output);
	void differentiate(int batchSize, int size, Matrix A, Matrix Ao, Matrix& AGrad, Matrix AoGrad);
	Activation* clone();
};

class Swish : public Activation {

public:
	void operate(int batchSize, int size, Matrix input, Matrix output);
	void differentiate(int batchSize, int size, Matrix A, Matrix Ao, Matrix& AGrad, Matrix AoGrad);
	Activation* clone();
};

class Softmax : public Activation {

public:
	void operate(int batchSize, int size, Matrix input, Matrix output);
	void differentiate(int batchSize, int size, Matrix A, Matrix Ao, Matrix& AGrad, Matrix AoGrad);
	Activation* clone();
	bool isDiagonal();
};

