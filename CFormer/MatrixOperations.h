#pragma once
#include "PropagationQueue.h"


class MUnary : public Operation {

public:
	Matrix2* A;
	int APrereq;
	bool ACopied;

	MUnary(Matrix2& A);
	void applyToStream(StreamEnvironment stream);
	void copyToDevice(StreamEnvironment stream, int completedIndex);
	void copyToHost(StreamEnvironment stream);
	void findPrereqs(vector<Operation*> operations);
};

class MBinary : public Operation {

public:
	Matrix2* A;
	Matrix2* B;
	int APrereq;
	bool ACopied;

	MBinary(Matrix2& A, Matrix2& B);
	void applyToStream(StreamEnvironment stream);
	void copyToDevice(StreamEnvironment stream, int completedIndex);
	void copyToHost(StreamEnvironment stream);
	void findPrereqs(vector<Operation*> operations);
};

class MMTrinary : public Operation {

public:
	Matrix2* A;
	Matrix2* B;
	Matrix2* C;
	int APrereq, BPrereq, CPrereq;
	bool ACopied, BCopied, CCopied;
	int outputLength;

	const float ALPHA = 1.0f;
	const float BETA0 = 0.0f;
	const float BETA1 = 1.0f;

	MMTrinary(Matrix2& A, Matrix2& B, Matrix2& C);
	void applyToStream(StreamEnvironment stream);
	void copyToDevice(StreamEnvironment stream, int completedIndex);
	void copyToHost(StreamEnvironment stream);
	void findPrereqs(vector<Operation*> operations);
};

class MConstantFill : MUnary {

public:
	float c;

	MConstantFill(Matrix2& A, float c) : MUnary(A) { this->c = c; };
	void operate(StreamEnvironment stream);
};

class MScale : MUnary {

public:
	float c;

	MScale(Matrix2& A, float c) : MUnary(A) { this->c = c; };
	void operate(StreamEnvironment stream);
};

class MMAdd : public MMTrinary {

public:
	MMAdd(Matrix2& A, Matrix2& B, Matrix2& C) : MMTrinary(A, B, C) {};
	void operate(StreamEnvironment stream);
};

class MMElementMultiply : public MMTrinary {

public:
	MMElementMultiply(Matrix2& A, Matrix2& B, Matrix2& C) : MMTrinary(A, B, C) {};
	void operate(StreamEnvironment stream);
};

class MMLinearCombo : public MMTrinary {

public:
	float c1, c2;

	MMLinearCombo(float c1, Matrix2& A, float c2, Matrix2& B, Matrix2& C) : MMTrinary(A, B, C) { this->c1 = c1; this->c2 = c2; };
	void operate(StreamEnvironment stream);
};

class MMMultiplyABC : public MMTrinary {

public:
	bool overwrite;

	MMMultiplyABC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) : MMTrinary(A, B, C) { this->overwrite = overwrite; };
	void operate(StreamEnvironment stream);
};

class MMMultiplyAtBC : public MMTrinary {

public:
	bool overwrite;

	MMMultiplyAtBC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) : MMTrinary(A, B, C) { this->overwrite = overwrite; };
	void operate(StreamEnvironment stream);
};

class MMMultiplyAtBtC : public MMTrinary {

public:
	bool overwrite;

	MMMultiplyAtBtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) : MMTrinary(A, B, C) { this->overwrite = overwrite; };
	void operate(StreamEnvironment stream);
};

class MMMultiplyABtC : public MMTrinary {

public:
	bool overwrite;

	MMMultiplyABtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite) : MMTrinary(A, B, C) { this->overwrite = overwrite; };
	void operate(StreamEnvironment stream);
};
