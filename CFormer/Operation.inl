template<typename Type>
HostToDeviceCopy<Type>::HostToDeviceCopy(GPUOperation* operation, Type* A, int deviceNum) {
	this->A = A;
	this->output = &A + 1;
	this->deviceNum;
	this->copyID = &operation->threadID;
	this->idFound = &operation->idFound;
	prereq = NULL;
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
int HostToDeviceCopy<Type>::getPrereqsUnmet() {
	return (prereq == NULL ? 0 : prereq->completed);
}

template<typename Type>
DeviceToHostCopy<Type>::DeviceToHostCopy(GPUOperation* operation, Type* A, int deviceNum) {
	this->A = A;
	this->output = A;
	this->deviceNum;
	this->copyID = &operation->threadID;
	this->idFound = &operation->idFound;
	prereq = NULL;
}

template<typename Type>
void DeviceToHostCopy<Type>::findPrereqs(vector<Operation*> operations, int index) {
	return;
}

template<typename Type>
int DeviceToHostCopy<Type>::getPrereqsUnmet() {
	return (prereq == NULL ? 0 : prereq->completed);
}

template<typename TypeA, typename TypeB, typename TypeC>
Trinary<TypeA, TypeB, TypeC>::Trinary(TypeA& A, TypeB& B, TypeC& C) {
	this->A = &A;
	this->B = &B;
	this->C = &C;
	this->output = &A + 1;
	this->prereqA = NULL;
	this->prereqB = NULL;
}

template<typename TypeA, typename TypeB, typename TypeC>
void Trinary<TypeA, TypeB, TypeC>::addDeviceCopies(vector<Operation*>& operations) {
	Operation* ACopy = new HostToDeviceCopy(this, this->A, 0);
	Operation* BCopy = new HostToDeviceCopy(this, this->B, 1);
	operations.emplace_back(ACopy);
	operations.emplace_back(BCopy);
	this->prereqA = ACopy;
	this->prereqB = BCopy;
}

template<typename TypeA, typename TypeB, typename TypeC>
int Multiply<TypeA, TypeB, TypeC>::getPrereqsUnmet() {
	return (this->prereqA == NULL ? 0 : this->prereqA->completed) + (this->prereqB == NULL ? 0 : this->prereqB->completed) + (this->prereqC == NULL ? 0 : this->prereqC->completed);
}

template<typename TypeA, typename TypeB, typename TypeC>
void Multiply<TypeA, TypeB, TypeC>::addDeviceCopies(vector<Operation*>& operations) {
	Operation* ACopy = new HostToDeviceCopy(this, this->A, 0);
	Operation* BCopy = new HostToDeviceCopy(this, this->B, 1);
	operations.emplace_back(ACopy);
	operations.emplace_back(BCopy);
	if (!this->overwrite) {
		Operation* CCopy = new HostToDeviceCopy(this, this->C, 2);
		operations.emplace_back(CCopy);
		this->prereqC = CCopy;
	}
	this->prereqA = ACopy;
	this->prereqB = BCopy;
}

template<typename TypeA, typename TypeB, typename TypeC>
void Trinary<TypeA, TypeB, TypeC>::addHostCopies(vector<Operation*>& operations) {
	DeviceToHostCopy<TypeC>* CCopy = new DeviceToHostCopy(this, this->C, 2);
	operations.emplace_back(CCopy);
	CCopy->prereq = this;
}