template<typename Type>
HUnary<Type>::HUnary(Type& A) {
	this->A = &A;
	this->numOutputs = 1;
	this->outputs = new void* [1] {&A};
	prereq = NULL;
}

template<typename Type>
void HUnary<Type>::findPrereqs(vector<Operation*> operations, int index) {
	for (int i = 0; i < index; i++) {
		for (int j = 0; j < operations[i]->numOutputs; j++) {
			if (operations[i]->outputs[j] == A) {
				prereq = operations[i];
				this->prereqDepth = operations[i]->prereqDepth + 1;
			}
		}
	}
}

template<typename Type>
int HUnary<Type>::getPrereqsUnmet(OperationQueue* queue) {
	return prereq == NULL ? 0 : prereq->completed.load();
}

template<typename TypeA, typename TypeB>
HBinary<TypeA, TypeB>::HBinary(TypeA& A, TypeB& B) {
	this->A = &A;
	this->B = &B;
	this->numOutputs = 1;
	this->outputs = new void* [1] {&B};
	prereqA = NULL;
}

template<typename TypeA, typename TypeB>
void HBinary<TypeA, TypeB>::findPrereqs(vector<Operation*> operations, int index) {
	for (int i = 0; i < index; i++) {
		for (int j = 0; j < operations[i]->numOutputs; j++) {
			if (operations[i]->outputs[j] == A) {
				prereqA = operations[i];
				this->prereqDepth = operations[i]->prereqDepth + 1;
			}
		}
	}
}

template<typename TypeA, typename TypeB>
int HBinary<TypeA, TypeB>::getPrereqsUnmet(OperationQueue* queue) {
	return prereqA == NULL ? 0 : prereqA->completed.load();
}

template<typename Type>
HostToDeviceCopy<Type>::HostToDeviceCopy(DOperation* operation, Type* A, int deviceNum, int batchSize) {
	this->A = A;
	this->numOutputs = 1;
	this->outputs = new void* [1] {NULL};
	this->deviceNum = deviceNum;
	this->parentOperation = operation;
	prereq = NULL;
	this->batchSize = batchSize;
}

template<typename Type>
void HostToDeviceCopy<Type>::findPrereqs(vector<Operation*> operations, int index) {
	for (int i = 0; i < index; i++) {
		for (int j = 0; j < operations[i]->numOutputs; j++) {
			if (operations[i]->outputs[j] == A) {
				prereq = operations[i];
				this->prereqDepth = operations[i]->prereqDepth + 1;
			}
		}
	}
}

template<typename Type>
int HostToDeviceCopy<Type>::getPrereqsUnmet(OperationQueue* queue) {
	int prereqSum = (prereq == NULL ? 0 : prereq->completed.load());
	int id = this->parentOperation->threadID.load();
	//if (id == -2 && this->A->isWeight) {
	//	return prereqSum + 1;
	//}
	if (this->prereqDepth != this->parentOperation->prereqDepth - 1) {
		return prereqSum + 1;
	} else if (id == -2 && queue->devicesUsed.load() >= queue->numThreads) {
		return prereqSum + 1;
	}
	else {
		return prereqSum;
	}
}

template<typename Type>
DeviceToHostCopy<Type>::DeviceToHostCopy(DOperation* operation, Type* A, int deviceNum) {
	this->A = A;
	this->numOutputs = 1;
	this->outputs = new void* [1] {A};
	this->deviceNum = deviceNum;
	this->numOperationOutputs = operation->numOutputs;
	this->parentOperation = operation;
	prereq = NULL;
}

template<typename Type>
void DeviceToHostCopy<Type>::findPrereqs(vector<Operation*> operations, int index) {
	this->prereqDepth = this->prereq == NULL ? 0 : (this->prereq->prereqDepth + 1);
	return;
}

template<typename Type>
int DeviceToHostCopy<Type>::getPrereqsUnmet(OperationQueue* queue) {
	return (prereq == NULL ? 0 : prereq->completed.load());
}

template<typename TypeA>
DUnary<TypeA>::DUnary(TypeA& A) {
	this->A = &A;
	this->numOutputs = 1;
	this->outputs = new void* [1] {NULL};
	prereq = NULL;
	this->outputsCopied.store(1);
}

template<typename TypeA>
int DUnary<TypeA>::getPrereqsUnmet(OperationQueue* queue) {
	return prereq == NULL ? 0 : prereq->completed.load();
}

template<typename TypeA>
void DUnary<TypeA>::addDeviceCopies(vector<Operation*>& operations) {
	Operation* ACopy = new HostToDeviceCopy(this, this->A, 0, typeid(TypeA) == typeid(MatrixBatch) ? ((MatrixBatch*)A)->batchSize : 1);
	operations.emplace_back(ACopy);
	prereq = ACopy;
}

template<typename TypeA>
void DUnary<TypeA>::addHostCopies(vector<Operation*>& operations) {
	DeviceToHostCopy<TypeA>* ACopy = new DeviceToHostCopy(this, this->A, 0);
	operations.emplace_back(ACopy);
	ACopy->prereq = this;
}

template<typename TypeA, typename TypeB>
DBinary<TypeA, TypeB>::DBinary(TypeA& A, TypeB& B) {
	this->A = &A;
	this->B = &B;
	this->numOutputs = 1;
	this->outputs = new void* [1] {NULL};
	prereqA = NULL;
	this->outputsCopied.store(1);
}

template<typename TypeA, typename TypeB>
int DBinary<TypeA, TypeB>::getPrereqsUnmet(OperationQueue* queue) {
	return prereqA == NULL ? 0 : prereqA->completed.load();
}

template<typename TypeA, typename TypeB>
void DBinary<TypeA, TypeB>::addDeviceCopies(vector<Operation*>& operations) {
	Operation* ACopy = new HostToDeviceCopy(this, this->A, 0, typeid(TypeB) == typeid(MatrixBatch) ? ((MatrixBatch*)B)->batchSize : 1);
	operations.emplace_back(ACopy);
	this->prereqA = ACopy;
}

template<typename TypeA, typename TypeB>
void DBinary<TypeA, TypeB>::addHostCopies(vector<Operation*>& operations) {
	DeviceToHostCopy<TypeB>* BCopy = new DeviceToHostCopy(this, this->B, 1);
	operations.emplace_back(BCopy);
	BCopy->prereq = this;
}

template<typename TypeA, typename TypeB, typename TypeC>
DTrinary<TypeA, TypeB, TypeC>::DTrinary(TypeA& A, TypeB& B, TypeC& C) {
	this->A = &A;
	this->B = &B;
	this->C = &C;
	this->numOutputs = 1;
	this->outputs = new void* [1] {&C};
	this->prereqA = NULL;
	this->prereqB = NULL;
	this->outputsCopied.store(1);
}

template<typename TypeA, typename TypeB, typename TypeC>
void DTrinary<TypeA, TypeB, TypeC>::addDeviceCopies(vector<Operation*>& operations) {
	Operation* ACopy = new HostToDeviceCopy(this, this->A, 0, typeid(TypeC) == typeid(MatrixBatch) ? ((MatrixBatch*)this->C)->batchSize : 1);
	Operation* BCopy = new HostToDeviceCopy(this, this->B, 1, typeid(TypeC) == typeid(MatrixBatch) ? ((MatrixBatch*)this->C)->batchSize : 1);
	operations.emplace_back(ACopy);
	operations.emplace_back(BCopy);
	this->prereqA = ACopy;
	this->prereqB = BCopy;
}

template<typename TypeA, typename TypeB, typename TypeC>
void DTrinary<TypeA, TypeB, TypeC>::addHostCopies(vector<Operation*>& operations) {
	DeviceToHostCopy<TypeC>* CCopy = new DeviceToHostCopy(this, this->C, 2);
	CCopy->prereq = this;
	operations.emplace_back(CCopy);
}

template<typename TypeA, typename TypeB, typename TypeC>
int DTrinary<TypeA, TypeB, TypeC>::getPrereqsUnmet(OperationQueue* queue) {
	return (prereqA == NULL ? 0 : prereqA->completed.load()) + (prereqB == NULL ? 0 : prereqB->completed.load());
}

template<typename TypeIn, typename TypeOut>
Dnary<TypeIn, TypeOut>::Dnary(int N_IN, int N_OUT) {
	this->N_IN = N_IN;
	this->N_OUT = N_OUT;
	this->in = new TypeIn * [N_IN];
	this->out = new TypeOut * [N_OUT];
	this->prereqs = new Operation * [N_IN];
	for (int i = 0; i < N_IN; i++) {
		this->prereqs[i] = NULL;
	}
	this->numOutputs = N_OUT;
	this->outputsCopied.store(N_OUT);
	this->outputs = (void**)this->out;
}

template<typename TypeIn, typename TypeOut>
int Dnary<TypeIn, TypeOut>::getPrereqsUnmet(OperationQueue* queue) {
	int sum = 0;
	for (int i = 0; i < N_IN; i++) {
		sum += prereqs[i] == NULL ? 0 : prereqs[i]->completed.load();
	}
	return sum;
}

template<typename TypeIn, typename TypeOut>
void Dnary<TypeIn, TypeOut>::addDeviceCopies(vector<Operation*>& operations) {
	for (int i = 0; i < N_IN; i++) {
		Operation* copy = new HostToDeviceCopy(this, this->in[i], i, 1);
		operations.emplace_back(copy);
		this->prereqs[i] = copy;
	}
}

template<typename TypeIn, typename TypeOut>
void Dnary<TypeIn, TypeOut>::addHostCopies(vector<Operation*>& operations) {
	for (int i = 0; i < N_OUT; i++) {
		DeviceToHostCopy<TypeOut>* copy = new DeviceToHostCopy(this, this->out[i], N_IN + i);
		copy->prereq = this;
		operations.emplace_back(copy);
	}
}

template<typename TypeA, typename TypeB, typename TypeC>
int Multiply<TypeA, TypeB, TypeC>::getPrereqsUnmet(OperationQueue* queue) {
	return (this->prereqA == NULL ? 0 : this->prereqA->completed.load()) + (this->prereqB == NULL ? 0 : this->prereqB->completed.load()) + (this->prereqC == NULL ? 0 : this->prereqC->completed.load());
}

template<typename TypeA, typename TypeB, typename TypeC>
void Multiply<TypeA, TypeB, TypeC>::addDeviceCopies(vector<Operation*>& operations) {
	Operation* ACopy = new HostToDeviceCopy(this, this->A, 0, typeid(TypeC) == typeid(MatrixBatch) ? ((MatrixBatch*)this->C)->batchSize : 1);
	Operation* BCopy = new HostToDeviceCopy(this, this->B, 1, typeid(TypeC) == typeid(MatrixBatch) ? ((MatrixBatch*)this->C)->batchSize : 1);
	operations.emplace_back(ACopy);
	operations.emplace_back(BCopy);
	if (!this->overwrite) {
		Operation* CCopy = new HostToDeviceCopy(this, this->C, 2, typeid(TypeC) == typeid(MatrixBatch) ? ((MatrixBatch*)this->C)->batchSize : 1);
		operations.emplace_back(CCopy);
		this->prereqC = CCopy;
	}
	this->prereqA = ACopy;
	this->prereqB = BCopy;
}

template<typename Type>
void DUnary<Type>::findPrereqs(vector<Operation*> operations, int index) {
	this->prereqDepth = this->prereq == NULL ? 0 : (this->prereq->prereqDepth + 1);
}

template<typename TypeA, typename TypeB>
void DBinary<TypeA, TypeB>::findPrereqs(vector<Operation*> operations, int index) {
	this->prereqDepth = this->prereqA == NULL ? 0 : (this->prereqA->prereqDepth + 1);
}

template<typename TypeA, typename TypeB, typename TypeC>
void DTrinary<TypeA, TypeB, TypeC>::findPrereqs(vector<Operation*> operations, int index) {
	this->prereqDepth = max(this->prereqA == NULL ? 0 : (this->prereqA->prereqDepth + 1), this->prereqB == NULL ? 0 : (this->prereqB->prereqDepth + 1));
}

template<typename TypeIn, typename TypeOut>
void Dnary<TypeIn, TypeOut>::findPrereqs(vector<Operation*> operations, int index) {
	this->prereqDepth = 0;
	for (int i = 0; i < this->N_IN; i++) {
		this->prereqDepth = max(this->prereqDepth, this->prereqs[i] == NULL ? 0 : (this->prereqs[i]->prereqDepth + 1));
	}
}

template<typename TypeA, typename TypeB, typename TypeC>
void Multiply<TypeA, TypeB, TypeC>::findPrereqs(vector<Operation*> operations, int index) {
	this->prereqDepth = max(this->prereqA == NULL ? 0 : (this->prereqA->prereqDepth + 1), this->prereqB == NULL ? 0 : (this->prereqB->prereqDepth + 1));
	this->prereqDepth = max(this->prereqDepth, this->prereqC == NULL ? 0 : (this->prereqC->prereqDepth + 1));
}