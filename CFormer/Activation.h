#pragma once
#include <cmath>
#include <fstream>
#include <climits>
#include "Matrix2.h"

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
		virtual void operate(Matrix2& input, Matrix2& output) = 0;
		virtual void operate(MatrixBatch& input, MatrixBatch& output) = 0;
		virtual void differentiate(Matrix2& input, Matrix2& output, Matrix2& inputGradient, Matrix2& outputGradient) = 0;
		virtual void differentiate(MatrixBatch& input, MatrixBatch& output, MatrixBatch& inputGradient, MatrixBatch& outputGradient) = 0;
		virtual Activation* clone() = 0;
		virtual bool isDiagonal() { return true; };
		virtual void save(ofstream& file) {
			string name(& typeid(*this).name()[6]);
			file << name << ",";
		};

};

class None : public Activation {

public:
	void operate(Matrix2& input, Matrix2& output);
	void operate(MatrixBatch& input, MatrixBatch& output);
	void differentiate(Matrix2& input, Matrix2& output, Matrix2& inputGradient, Matrix2& outputGradient);
	void differentiate(MatrixBatch& input, MatrixBatch& output, MatrixBatch& inputGradient, MatrixBatch& outputGradient);
	Activation* clone();
};

class Sigmoid : public Activation {

public:
	void operate(Matrix2& input, Matrix2& output);
	void operate(MatrixBatch& input, MatrixBatch& output);
	void differentiate(Matrix2& input, Matrix2& output, Matrix2& inputGradient, Matrix2& outputGradient);
	void differentiate(MatrixBatch& input, MatrixBatch& output, MatrixBatch& inputGradient, MatrixBatch& outputGradient);
	Activation* clone();
};

class Relu : public Activation {

public:
	void operate(Matrix2& input, Matrix2& output);
	void operate(MatrixBatch& input, MatrixBatch& output);
	void differentiate(Matrix2& input, Matrix2& output, Matrix2& inputGradient, Matrix2& outputGradient);
	void differentiate(MatrixBatch& input, MatrixBatch& output, MatrixBatch& inputGradient, MatrixBatch& outputGradient);
	Activation* clone();
};

class Elu : public Activation {

public:
	float alpha;
	Elu(float alpha);
	void operate(Matrix2& input, Matrix2& output);
	void operate(MatrixBatch& input, MatrixBatch& output);
	void differentiate(Matrix2& input, Matrix2& output, Matrix2& inputGradient, Matrix2& outputGradient);
	void differentiate(MatrixBatch& input, MatrixBatch& output, MatrixBatch& inputGradient, MatrixBatch& outputGradient);
	Activation* clone();
	void save(ofstream& file);
};

class Selu : public Activation {
public:
	void operate(Matrix2& input, Matrix2& output);
	void operate(MatrixBatch& input, MatrixBatch& output);
	void differentiate(Matrix2& input, Matrix2& output, Matrix2& inputGradient, Matrix2& outputGradient);
	void differentiate(MatrixBatch& input, MatrixBatch& output, MatrixBatch& inputGradient, MatrixBatch& outputGradient);
	Activation* clone();
};

class Tanh : public Activation {

public:
	void operate(Matrix2& input, Matrix2& output);
	void operate(MatrixBatch& input, MatrixBatch& output);
	void differentiate(Matrix2& input, Matrix2& output, Matrix2& inputGradient, Matrix2& outputGradient);
	void differentiate(MatrixBatch& input, MatrixBatch& output, MatrixBatch& inputGradient, MatrixBatch& outputGradient);
	Activation* clone();
};

class Swish : public Activation {

public:
	void operate(Matrix2& input, Matrix2& output);
	void operate(MatrixBatch& input, MatrixBatch& output);
	void differentiate(Matrix2& input, Matrix2& output, Matrix2& inputGradient, Matrix2& outputGradient);
	void differentiate(MatrixBatch& input, MatrixBatch& output, MatrixBatch& inputGradient, MatrixBatch& outputGradient);
	Activation* clone();
};

class Softmax : public Activation {

public:
	void operate(Matrix2& input, Matrix2& output);
	void operate(MatrixBatch& input, MatrixBatch& output);
	void differentiate(Matrix2& input, Matrix2& output, Matrix2& inputGradient, Matrix2& outputGradient);
	void differentiate(MatrixBatch& input, MatrixBatch& output, MatrixBatch& inputGradient, MatrixBatch& outputGradient);
	Activation* clone();
	bool isDiagonal();
};

