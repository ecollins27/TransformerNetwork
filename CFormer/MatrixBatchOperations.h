#pragma once
#include "PropagationQueue.h"

class BBTrinary : public Operation {

public:
	MatrixBatch* A;
	MatrixBatch* B;
	MatrixBatch* C;
	int outputLength;

	const float ALPHA = 1.0f;
	const float BETA0 = 0.0f;
	const float BETA1 = 1.0f;

	BBTrinary(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C);
	void applyToStream(StreamEnvironment stream);
	void copyToDevice(StreamEnvironment stream);
	void copyToHost(StreamEnvironment stream);
};

class MBTrinary : public Operation {

public:
	Matrix2* A;
	MatrixBatch* B;
	MatrixBatch* C;
	int outputLength;

	const float ALPHA = 1.0f;
	const float BETA0 = 0.0f;
	const float BETA1 = 1.0f;

	MBTrinary(Matrix2& A, MatrixBatch& B, MatrixBatch& C);
	void applyToStream(StreamEnvironment stream);
	void copyToDevice(StreamEnvironment stream);
	void copyToHost(StreamEnvironment stream);
};

class BBAdd : public BBTrinary {

public:
	BBAdd(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C) : BBTrinary(A, B, C) {};
	void operate(StreamEnvironment stream);
};

class BBElementMultiply : public BBTrinary {

public:
	BBElementMultiply(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C) : BBTrinary(A, B, C) {};
	void operate(StreamEnvironment stream);
};

class BBLinearCombo : public BBTrinary {

public:
	float c1, c2;

	BBLinearCombo(float c1, MatrixBatch& A, float c2, MatrixBatch& B, MatrixBatch& C) : BBTrinary(A, B, C) { this->c1 = c1; this->c2 = c2; };
	void operate(StreamEnvironment stream);
};

class BBMultiplyABC : public BBTrinary {

public:
	bool overwrite;

	BBMultiplyABC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) : BBTrinary(A, B, C) { this->overwrite = overwrite; };
	void operate(StreamEnvironment stream);
};

class BBMultiplyAtBC : public BBTrinary {

public:
	bool overwrite;

	BBMultiplyAtBC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) : BBTrinary(A, B, C) { this->overwrite = overwrite; };
	void operate(StreamEnvironment stream);
};

class BBMultiplyAtBtC : public BBTrinary {

public:
	bool overwrite;

	BBMultiplyAtBtC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) : BBTrinary(A, B, C) { this->overwrite = overwrite; };
	void operate(StreamEnvironment stream);
};

class BBMultiplyABtC : public BBTrinary {

public:
	bool overwrite;

	BBMultiplyABtC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite) : BBTrinary(A, B, C) { this->overwrite = overwrite; };
	void operate(StreamEnvironment stream);
};
