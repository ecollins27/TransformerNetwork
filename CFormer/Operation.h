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

class DOperation : public Operation {

public:
	atomic<int> threadID;

	virtual void applyToStream(PropagationQueue* queue);
	virtual void findPrereqs(vector<Operation*> operations, int index);
	virtual void addDeviceCopies(vector<Operation*>& operations) = 0;
	virtual void addHostCopies(vector<Operation*>& operations) = 0;
};

class HOperation : public Operation {

public:
	virtual void applyToStream(PropagationQueue* queue) { return; };
};

template<typename Type>
class HUnary : public HOperation {

public:
	Type* A;
	Operation* prereq;

	HUnary(Type& A);
	virtual void findPrereqs(vector<Operation*> operations, int index);
	virtual int getPrereqsUnmet(PropagationQueue* queue);
	virtual bool containsPrereq(Operation* o);
};

template<typename TypeA, typename TypeB>
class HBinary : public HOperation {

public:
	TypeA* A;
	TypeB* B;
	Operation* prereqA;

	HBinary(TypeA& A, TypeB& B);
	virtual void findPrereqs(vector<Operation*> operations, int index);
	virtual int getPrereqsUnmet(PropagationQueue* queue);
	virtual bool containsPrereq(Operation* o);
};

template<typename Type>
class HostToDeviceCopy : public Operation {

public:
	Type* A;
	int deviceNum;
	atomic<int>* threadID;
	Operation* prereq;
	int batchSize;

	HostToDeviceCopy(DOperation* operation, Type* A, int deviceNum, int batchSize);
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

	DeviceToHostCopy(DOperation* operation, Type* A, int deviceNum);
	bool operate(PropagationQueue* queue, int threadID);
	void applyToStream(PropagationQueue* queue);
	void findPrereqs(vector<Operation*> operations, int index);
	bool containsPrereq(Operation* o);
	int getPrereqsUnmet(PropagationQueue* queue);
};

template<typename TypeA>
class DUnary : public DOperation {

public:
	TypeA* A;
	Operation* prereq;

	DUnary(TypeA& A);
	virtual int getPrereqsUnmet(PropagationQueue* queue);
	virtual void addDeviceCopies(vector<Operation*>& operations);
	virtual void addHostCopies(vector<Operation*>& operations);
	virtual bool containsPrereq(Operation* o);
};

template<typename TypeA, typename TypeB>
class DBinary : public DOperation {

public:
	TypeA* A;
	TypeB* B;
	Operation* prereqA;

	DBinary(TypeA& A, TypeB& B);
	virtual int getPrereqsUnmet(PropagationQueue* queue);
	virtual void addDeviceCopies(vector<Operation*>& operations);
	virtual void addHostCopies(vector<Operation*>& operations);
	virtual bool containsPrereq(Operation* o);
};

template<typename TypeA, typename TypeB, typename TypeC>
class DTrinary : public DOperation {

public:
	TypeA* A;
	TypeB* B;
	TypeC* C;
	Operation* prereqA;
	Operation* prereqB;

	DTrinary(TypeA& A, TypeB& B, TypeC& C);
	virtual int getPrereqsUnmet(PropagationQueue* queue);
	virtual void addDeviceCopies(vector<Operation*>& operations);
	virtual void addHostCopies(vector<Operation*>& operations);
	virtual bool containsPrereq(Operation* o);
};

template<typename TypeA, typename TypeB, typename TypeC>
class Multiply : public DTrinary<TypeA, TypeB, TypeC> {

public:
	bool overwrite;
	Operation* prereqC;
	const float ALPHA = 1.0f;
	const float BETA0 = 0.0f;
	const float BETA1 = 1.0f;

	Multiply(TypeA& A, TypeB& B, TypeC& C, bool overwrite) : DTrinary<TypeA, TypeB, TypeC>(A, B, C) { this->overwrite = overwrite; this->prereqC = NULL; };
	int getPrereqsUnmet(PropagationQueue* queue);
	virtual void addDeviceCopies(vector<Operation*>& operations);
	virtual bool containsPrereq(Operation* o);
};

template<typename Type>
class Print : public HUnary<Type> {

public:
	Print(Type& A) : HUnary<Type>(A) { return; };
	bool operate(PropagationQueue* queue, int threadID);
};

template<typename Type>
class CopyTo : public HBinary<Type, Type> {

public:
	int customWidth;
	CopyTo(Type& A, Type& B) : HBinary<Type, Type>(A, B) { this->customWidth = -1; };
	CopyTo(int width, Type& A, Type& B) : HBinary<Type, Type>(A, B) { this->customWidth = width; };
	bool operate(PropagationQueue* queue, int threadID);
};

template<typename Type>
class ConstantFill : public DUnary<Type> {

public:
	float c;
	ConstantFill(Type& A, float c) : DUnary<Type>(A) { this->c = c; };
	bool operate(PropagationQueue* queue, int threadID);
};

template<typename Type>
class Scale : public DUnary<Type> {

public:
	float c;
	Scale(Type& A, float c) : DUnary<Type>(A) { this->c = c; };
	bool operate(PropagationQueue* queue, int threadID);
};

template<typename Type>
class Sqrt : public DBinary<Type, Type> {

public:
	Sqrt(Type& A, Type& B) : DBinary<Type, Type>(A, B) { return; };
	bool operate(PropagationQueue* queue, int threadID);
};

template<typename Type>
class Transpose : public DBinary<Type, Type> {

public:
	Transpose(Type& A, Type& B) : DBinary<Type, Type>(A, B) { return; };
	bool operate(PropagationQueue* queue, int threadID);
};

class Condense : public DBinary<MatrixBatch, Matrix2> {

public:
	Condense(MatrixBatch& A, Matrix2& B) : DBinary<MatrixBatch, Matrix2>(A, B) { return; };
	bool operate(PropagationQueue* queue, int threadID);
};

template<typename Type>
class Add : public DTrinary<Type, Type, Type> {

public:
	int customWidth;
	Add(Type& A, Type& B, Type& C) : DTrinary<Type, Type, Type>(A, B, C) { this->customWidth = -1; };
	Add(int width, Type& A, Type& B, Type& C) : DTrinary<Type, Type, Type>(A, B, C) { this->customWidth = width; };
	bool operate(PropagationQueue* queue, int threadID);
};

template<typename Type>
class ElementMultiply : public DTrinary<Type, Type, Type> {

public:
	ElementMultiply(Type& A, Type& B, Type& C) : DTrinary<Type, Type, Type>(A, B, C) { return; };
	bool operate(PropagationQueue* queue, int threadID);
};

template<typename Type>
class LinearCombo : public DTrinary<Type, Type, Type> {

public:
	float c1, c2;
	LinearCombo(float c1, Type& A, float c2, Type& B, Type& C) : DTrinary<Type, Type, Type>(A, B, C) { this->c1 = c1; this->c2 = c2; };
	bool operate(PropagationQueue* queue, int threadID);
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
	MultiplyAtBC(TypeA& A, TypeB& B, TypeC& C, bool overwrite) : Multiply<TypeA, TypeB, TypeC>(A, B, C, overwrite) { return; };
	bool operate(PropagationQueue* queue, int threadID);
};

template<typename TypeA, typename TypeB, typename TypeC>
class MultiplyAtBtC : public Multiply<TypeA, TypeB, TypeC> {

public:
	MultiplyAtBtC(TypeA& A, TypeB& B, TypeC& C, bool overwrite) : Multiply<TypeA, TypeB, TypeC>(A, B, C, overwrite) { return; };
	bool operate(PropagationQueue* queue, int threadID);
};

template<typename TypeA, typename TypeB, typename TypeC>
class MultiplyABtC : public Multiply<TypeA, TypeB, TypeC> {

public:
	MultiplyABtC(TypeA& A, TypeB& B, TypeC& C, bool overwrite) : Multiply<TypeA, TypeB, TypeC>(A, B, C, overwrite) { return; };
	bool operate(PropagationQueue* queue, int threadID);
};

#include "Operation.inl"