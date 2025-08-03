template<typename Type>
HUnary<Type>::HUnary(Type& A) {
	this->A = &A;
	this->output = &A;
	prereq = NULL;
}

template<typename Type>
void HUnary<Type>::findPrereqs(vector<Operation*> operations, int index) {
	for (int i = 0; i < index; i++) {
		if (operations[i]->output == A) {
			prereq = operations[i];
		}
	}
}

template<typename Type>
int HUnary<Type>::getPrereqsUnmet(PropagationQueue* queue) {
	return prereq->completed.load();
}

template<typename Type>
bool HUnary<Type>::containsPrereq(Operation* o) {
	return this->prereq == o;
}

template<typename TypeA, typename TypeB>
HBinary<TypeA, TypeB>::HBinary(TypeA& A, TypeB& B) {
	this->A = &A;
	this->B = &B;
	this->output = &B;
	prereqA = NULL;
}

template<typename TypeA, typename TypeB>
void HBinary<TypeA, TypeB>::findPrereqs(vector<Operation*> operations, int index) {
	for (int i = 0; i < index; i++) {
		if (operations[i]->output == A) {
			prereqA = operations[i];
		}
	}
}

template<typename TypeA, typename TypeB>
int HBinary<TypeA, TypeB>::getPrereqsUnmet(PropagationQueue* queue) {
	return prereqA->completed.load();
}

template<typename TypeA, typename TypeB>
bool HBinary<TypeA, TypeB>::containsPrereq(Operation* o) {
	return this->prereqA == o;
}

template<typename Type>
HostToDeviceCopy<Type>::HostToDeviceCopy(DOperation* operation, Type* A, int deviceNum, int batchSize) {
	this->A = A;
	this->output = A + 1;
	this->deviceNum = deviceNum;
	this->threadID = &operation->threadID;
	prereq = NULL;
	this->batchSize = batchSize;
}

template<typename Type>
void HostToDeviceCopy<Type>::findPrereqs(vector<Operation*> operations, int index) {
	for (int i = 0; i < index; i++) {
		if (operations[i]->output == A) {
			prereq = operations[i];
		}
	}
}

template<typename Type>
int HostToDeviceCopy<Type>::getPrereqsUnmet(PropagationQueue* queue) {
	return (prereq == NULL ? 0 : prereq->completed.load()) + ((this->threadID->load() == -1 && queue->devicesUsed >= queue->numThreads) ? 1 : 0);
}

template<typename Type>
bool HostToDeviceCopy<Type>::containsPrereq(Operation* o) {
	return this->prereq == o;
}

template<typename Type>
DeviceToHostCopy<Type>::DeviceToHostCopy(DOperation* operation, Type* A, int deviceNum) {
	this->A = A;
	this->output = A;
	this->deviceNum = deviceNum;
	this->threadID = &operation->threadID;
	prereq = NULL;
}

template<typename Type>
void DeviceToHostCopy<Type>::findPrereqs(vector<Operation*> operations, int index) {
	return;
}

template<typename Type>
int DeviceToHostCopy<Type>::getPrereqsUnmet(PropagationQueue* queue) {
	return (prereq == NULL ? 0 : prereq->completed.load());
}

template<typename Type>
bool DeviceToHostCopy<Type>::containsPrereq(Operation* o) {
	return this->prereq == o;
}

template<typename TypeA>
DUnary<TypeA>::DUnary(TypeA& A) {
	this->A = &A;
	this->output = &A + 1;
	prereq = NULL;
}

template<typename TypeA>
int DUnary<TypeA>::getPrereqsUnmet(PropagationQueue* queue) {
	return prereq->completed.load();
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

template<typename TypeA>
bool DUnary<TypeA>::containsPrereq(Operation* o) {
	return this->prereq == o;
}

template<typename TypeA, typename TypeB>
DBinary<TypeA, TypeB>::DBinary(TypeA& A, TypeB& B) {
	this->A = &A;
	this->B = &B;
	this->output = &B + 1;
	prereqA = NULL;
}

template<typename TypeA, typename TypeB>
int DBinary<TypeA, TypeB>::getPrereqsUnmet(PropagationQueue* queue) {
	return prereqA->completed.load();
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

template<typename TypeA, typename TypeB>
bool DBinary<TypeA, TypeB>::containsPrereq(Operation* o) {
	return this->prereqA == o;
}

template<typename TypeA, typename TypeB, typename TypeC>
DTrinary<TypeA, TypeB, TypeC>::DTrinary(TypeA& A, TypeB& B, TypeC& C) {
	this->A = &A;
	this->B = &B;
	this->C = &C;
	this->output = &A + 1;
	this->prereqA = NULL;
	this->prereqB = NULL;
}

template<typename TypeA, typename TypeB, typename TypeC>
void DTrinary<TypeA, TypeB, TypeC>::addDeviceCopies(vector<Operation*>& operations) {
	Operation* ACopy = new HostToDeviceCopy(this, this->A, 0, typeid(TypeC) == typeid(MatrixBatch)? ((MatrixBatch*)this->C)->batchSize : 1);
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
int DTrinary<TypeA, TypeB, TypeC>::getPrereqsUnmet(PropagationQueue* queue) {
	return (prereqA == NULL ? 0 : prereqA->completed.load()) + (prereqB == NULL ? 0 : prereqB->completed.load());
}

template<typename TypeA, typename TypeB, typename TypeC>
bool DTrinary<TypeA, TypeB, TypeC>::containsPrereq(Operation* o) {
	return this->prereqA == o || this->prereqB == o;
}

template<typename TypeA, typename TypeB, typename TypeC>
int Multiply<TypeA, TypeB, TypeC>::getPrereqsUnmet(PropagationQueue* queue) {
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

template<typename TypeA, typename TypeB, typename TypeC>
bool Multiply<TypeA, TypeB, TypeC>::containsPrereq(Operation* o) {
	return this->prereqA == o || this->prereqB == o || this->prereqC == o;
}