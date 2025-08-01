#pragma once
#include "Matrix2.h"
#include "MatrixBatch.h"
#include <atomic>

class PropagationQueue;

class Operation {

public:
	void* output;
	atomic<int> completed;
	atomic<bool> operationAllocated;

	virtual bool operate(PropagationQueue* queue, int threadID) = 0;
	virtual void applyToStream(PropagationQueue* queue) = 0;
	virtual void findPrereqs(vector<Operation*> operations, int index) = 0;
	virtual int getPrereqsUnmet(PropagationQueue* queue) = 0;
	virtual bool containsPrereq(Operation* o) = 0;
};

class GPUOperation : public Operation {

public:
	atomic<int> threadID;

	virtual void applyToStream(PropagationQueue* queue);
	virtual void findPrereqs(vector<Operation*> operations, int index);
	virtual void addDeviceCopies(vector<Operation*>& operations) = 0;
	virtual void addHostCopies(vector<Operation*>& operations) = 0;
};

template<typename Type>
class HostToDeviceCopy : public Operation {

public:
	Type* A;
	int deviceNum;
	atomic<int>* threadID;
	Operation* prereq;

	HostToDeviceCopy(GPUOperation* operation, Type* A, int deviceNum);
	bool operate(PropagationQueue* queue, int threadID);
	void applyToStream(PropagationQueue* queue);
	void findPrereqs(vector<Operation*> operations, int index);
	bool containsPrereq(Operation* o);
	int getPrereqsUnmet(PropagationQueue* queue);
};

template<typename Type>
class DeviceToHostCopy : public Operation {

public:
	Type* A;
	int deviceNum;
	atomic<int>* threadID;
	Operation* prereq;

	DeviceToHostCopy(GPUOperation* operation, Type* A, int deviceNum);
	bool operate(PropagationQueue* queue, int threadID);
	void applyToStream(PropagationQueue* queue);
	void findPrereqs(vector<Operation*> operations, int index);
	bool containsPrereq(Operation* o);
	int getPrereqsUnmet(PropagationQueue* queue);
};

template<typename TypeA>
class Unary : public GPUOperation {

public:
	TypeA* A;
	Operation* prereq;

	Unary(TypeA& A);
	virtual int getPrereqsUnmet(PropagationQueue* queue);
	virtual void addDeviceCopies(vector<Operation*>& operations);
	virtual void addHostCopies(vector<Operation*>& operations);
	virtual bool containsPrereq(Operation* o);
};

template<typename TypeA, typename TypeB>
class Binary : public GPUOperation {

public:
	TypeA* A;
	TypeB* B;
	Operation* prereqA;

	Binary(TypeA& A, TypeB& B);
	virtual int getPrereqsUnmet(PropagationQueue* queue);
	virtual void addDeviceCopies(vector<Operation*>& operations);
	virtual void addHostCopies(vector<Operation*>& operations);
	virtual bool containsPrereq(Operation* o);
};

template<typename TypeA, typename TypeB, typename TypeC>
class Trinary : public GPUOperation {

public:
	TypeA* A;
	TypeB* B;
	TypeC* C;
	Operation* prereqA;
	Operation* prereqB;

	Trinary(TypeA& A, TypeB& B, TypeC& C);
	virtual int getPrereqsUnmet(PropagationQueue* queue);
	virtual void addDeviceCopies(vector<Operation*>& operations);
	virtual void addHostCopies(vector<Operation*>& operations);
	virtual bool containsPrereq(Operation* o);
};

template<typename TypeA, typename TypeB, typename TypeC>
class Multiply : public Trinary<TypeA, TypeB, TypeC> {

public:
	bool overwrite;
	Operation* prereqC;
	const float ALPHA = 1.0f;
	const float BETA0 = 0.0f;
	const float BETA1 = 1.0f;

	Multiply(TypeA& A, TypeB& B, TypeC& C, bool overwrite) : Trinary<TypeA, TypeB, TypeC>(A, B, C) { this->overwrite = overwrite; this->prereqC = NULL; };
	int getPrereqsUnmet(PropagationQueue* queue);
	virtual void addDeviceCopies(vector<Operation*>& operations);
	virtual bool containsPrereq(Operation* o);
};

template<typename TypeA, typename TypeB, typename TypeC>
class MultiplyABC : public Multiply<TypeA, TypeB, TypeC> {

public:
	MultiplyABC(TypeA& A, TypeB& B, TypeC& C, bool overwrite) : Multiply<TypeA, TypeB, TypeC>(A, B, C, overwrite) { return; };
	bool operate(PropagationQueue* queue, int threadID);
};

template<typename TypeA, typename TypeB, typename TypeC>
class MultiplyAtBC : public Multiply<TypeA, TypeB, TypeC> {

public:

	bool operate(PropagationQueue* queue, int threadID);
};

template<typename TypeA, typename TypeB, typename TypeC>
class MultiplyAtBtC : public Multiply<TypeA, TypeB, TypeC> {

public:

	bool operate(PropagationQueue* queue, int threadID);
};

template<typename TypeA, typename TypeB, typename TypeC>
class MultiplyABtC : public Multiply<TypeA, TypeB, TypeC> {

public:

	bool operate(PropagationQueue* queue, int threadID);
};

#include "Operation.inl"