#include "LinformerAttention.h"
#include "Model.h"
#include "ModelParser.h"

const string LinformerAttention::LAYER_NAME = "LinformerAttention";

LinformerAttention::LinformerAttention(int numHeads, int keySize, int valueSize, int projSize) {
	this->numHeads = numHeads;
	this->keySize = keySize;
	this->valueSize = valueSize;
	this->projSize = projSize;
	softmax = Activation::SOFTMAX->clone();
}

void LinformerAttention::propagateLayer(int num) {
	double scalar = 1.0 / sqrt(keySize);
	for (int i = 0; i < numHeads; i++) {
		// Q = X WqT
		Matrix::multiplyABtC(numTokens[num], prevSize, keySize, prevLayer->neurons[num], Wq[i], Q[num][i], true);
		// K = Wk XT
		Matrix::multiplyABtC(keySize, prevSize, numTokens[num], Wk[i], prevLayer->neurons[num], K[num][i], true);
		// Kp = E KT
		Matrix::multiplyABtC(projSize, numTokens[num], keySize, E, K[num][i], KProj[num][i], true);
		// A = Q KT
		Matrix::multiplyABtC(numTokens[num], keySize, projSize, Q[num][i], K[num][i], A[num][i], true);
		// A = A / sqrt(dk)
		A[num][i].scale(numTokens[num], projSize, scalar);
		// A = softmax(A)
		softmax->operate(numTokens[num], projSize, A[num][i], A[num][i]);
		// V = Wv XT
		Matrix::multiplyABtC(valueSize, prevSize, numTokens[num], Wv[i], prevLayer->neurons[num], V[num][i], true);
		// Vp = V FT
		Matrix::multiplyABtC(valueSize, numTokens[num], projSize, V[num][i], F, VProj[num][i], true);
		// Ac = A VT
		Matrix::multiplyABtC(numTokens[num], projSize, valueSize, A[num][i], V[num][i], AcSub[num][i], true);
	}
	// Y = Ac WoT
	Matrix::multiplyABtC(numTokens[num], numHeads * valueSize, size, Ac[num], Wo, neurons[num], true);
}

void LinformerAttention::backPropagate(int num) {
	// dAc = dY Wo
	Matrix::multiplyABC(numTokens[num], size, numHeads * valueSize, neuronGradient[num], Wo, AcGrad[num], true);
	// dWo = dYT Ac
	Matrix::multiplyAtBC(size, numTokens[num], numHeads * valueSize, neuronGradient[num], Ac[num], WoGrad[num], true);
	prevLayer->neuronGradient[num].constantFill(0, numTokens[num], prevSize);
	EGrad->constantFill(0, projSize, maxNumTokens);
	FGrad->constantFill(0, projSize, maxNumTokens);
	float scalar = 1.0 / sqrt(keySize);
	for (int i = 0; i < numHeads; i++) {
		// dA = dAc Vp
		Matrix::multiplyABC(numTokens[num], valueSize, projSize, AcSubGrad[num][i], VProj[num][i], AGrad[num][i], true);
		// dVp = dAcT A
		Matrix::multiplyAtBC(valueSize, numTokens[num], projSize, AcSubGrad[num][i], A[num][i], VProjGrad[num][i], true);
		// dV = dVp F
		Matrix::multiplyABC(valueSize, projSize, numTokens[num], VProjGrad[num][i], F, VGrad[num][i], true);
		// dWv = dV X
		Matrix::multiplyABC(valueSize, numTokens[num], prevSize, VGrad[num][i], prevLayer->neurons[num], WvGrad[num][i], true);
		// dX += dVT Wv
		Matrix::multiplyAtBC(numTokens[num], valueSize, prevSize, VGrad[num][i], Wv[i], prevLayer->neuronGradient[num], false);
		// dF += dVpT V
		Matrix::multiplyAtBC(projSize, valueSize, numTokens[num], VProjGrad[num][i], V[num][i], FGrad[num], false);
		// dA = softmax'(A)
		softmax->differentiate(numTokens[num], projSize, A[num][i], A[num][i], AGrad[num][i], AGrad[num][i]);
		// dQ = dA Kp
		Matrix::multiplyABC(numTokens[num], projSize, keySize, AGrad[num][i], KProj[num][i], QGrad[num][i], true);
		// dQ = dQ / sqrt(dk)
		QGrad[num][i].scale(numTokens[num], keySize, scalar);
		// dKp = dAT Q
		Matrix::multiplyAtBC(projSize, numTokens[num], keySize, AGrad[num][i], Q[num][i], KProjGrad[num][i], true);
		// dKp = dKp / sqrt(dk)
		KProjGrad[num][i].scale(projSize, keySize, scalar);
		// dE += dKp K
		Matrix::multiplyABC(projSize, keySize, numTokens[num], KProjGrad[num][i], K[num][i], EGrad[num], false);
		// dK = dKpT E
		Matrix::multiplyAtBC(keySize, projSize, numTokens[num], KProjGrad[num][i], E, KGrad[num][i], true);
		//dX += dQ Wq
		Matrix::multiplyABC(numTokens[num], keySize, prevSize, QGrad[num][i], Wq[i], prevLayer->neuronGradient[num], false);
		// dWq += dQT X
		Matrix::multiplyAtBC(keySize, numTokens[num], prevSize, QGrad[num][i], prevLayer->neurons[num], WqGrad[num][i], true);
		// dWk = dK X
		Matrix::multiplyABC(keySize, numTokens[num], prevSize, KGrad[num][i], prevLayer->neurons[num], WkGrad[num][i], true);
		// dX += dKT Wk
		Matrix::multiplyAtBC(numTokens[num], keySize, prevSize, KGrad[num][i], Wk[i], prevLayer->neuronGradient[num], false);
	}
	prevLayer->backPropagate(num);
}

void LinformerAttention::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
	size = prevLayer->size;
	prevSize = prevLayer->size + 1;
	float std = 1.0 / size;
	Matrix::NormalFill* normal = new Matrix::NormalFill(0, std);
	Wq = Matrix::allocateMatrixArray(normal, numHeads, keySize, prevSize, true);
	Wk = Matrix::allocateMatrixArray(normal, numHeads, keySize, prevSize, true);
	Wv = Matrix::allocateMatrixArray(normal, numHeads, valueSize, prevSize, true);
	Wo = Matrix(normal, size, numHeads * valueSize, true);
}

void LinformerAttention::setBatchSize(int batchSize) {
	Layer2D::initNeurons(batchSize);
	WqGrad = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, keySize, prevSize, false);
	WkGrad = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, keySize, prevSize, false);
	WvGrad = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, valueSize, prevSize, false);
	WoGrad = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, size, numHeads * valueSize, false);
	EGrad = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, projSize, maxNumTokens, true);
	FGrad = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, projSize, maxNumTokens, true);
	K = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, keySize, maxNumTokens, true);
	KGrad = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, keySize, maxNumTokens, true);
	KProj = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, projSize, keySize, true);
	KProjGrad = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, projSize, keySize, true);
	Q = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, maxNumTokens, keySize, true);
	QGrad = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, maxNumTokens, keySize, true);
	V = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, valueSize, maxNumTokens, true);
	VGrad = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, valueSize, maxNumTokens, true);
	VProj = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, valueSize, projSize, true);
	VProjGrad = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, valueSize, projSize, true);
	A = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, maxNumTokens, projSize, true);
	AGrad = Matrix::allocateMatrixArray2D(Matrix::ZERO_FILL, batchSize, numHeads, maxNumTokens, projSize, true);
	Ac = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, maxNumTokens, numHeads * valueSize, true);
	AcGrad = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, maxNumTokens, numHeads * valueSize, true);
	AcSub = new Matrix * [batchSize];
	AcSubGrad = new Matrix * [batchSize];
	for (int i = 0; i < batchSize; i++) {
		AcSub[i] = new Matrix[numHeads];
		AcSubGrad[i] = new Matrix[numHeads];
		for (int j = 0; j < numHeads; j++) {
			AcSub[i][j] = Ac[i].subMatrix(0, j * valueSize, maxNumTokens, valueSize);
			AcSubGrad[i][j] = AcGrad[i].subMatrix(0, j * valueSize, maxNumTokens, valueSize);
		}
	}
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void LinformerAttention::save(ofstream& file) {
	file << LAYER_NAME << ", ";
	file << numHeads << "," << keySize << "," << valueSize << "," << projSize << ",\n";
	for (int i = 0; i < numHeads; i++) {
		for (int j = 0; j < keySize; j++) {
			for (int k = 0; k < prevSize; k++) {
				file << Wq[i](j, k) << ",";
			}
			file << "\n";
		}
		for (int j = 0; j < keySize; j++) {
			for (int k = 0; k < prevSize; k++) {
				file << Wk[i](j, k) << ",";
			}
			file << "\n";
		}
		for (int j = 0; j < valueSize; j++) {
			for (int k = 0; k < prevSize; k++) {
				file << Wv[i](j, k) << ",";
			}
			file << "\n";
		}
	}
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < numHeads * valueSize; j++) {
			file << Wo(i, j) << ",";
		}
		file << "\n";
	}
	file << maxNumTokens << ",\n";
	for (int i = 0; i < projSize; i++) {
		for (int j = 0; j < maxNumTokens; j++) {
			file << E(i, j) << ",";
		}
		file << "\n";
	}
	for (int i = 0; i < projSize; i++) {
		for (int j = 0; j < maxNumTokens; j++) {
			file << F(i, j) << ",";
		}
		file << "\n";
	}
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void LinformerAttention::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	int numHeads = ModelParser::getNextInt(line, commaIndex, newCommaIndex);
	int keySize = ModelParser::getNextInt(line, commaIndex, newCommaIndex);
	int valueSize = ModelParser::getNextInt(line, commaIndex, newCommaIndex);
	int projSize = ModelParser::getNextInt(line, commaIndex, newCommaIndex);
	LinformerAttention* linformerAttention = { new LinformerAttention(numHeads, keySize, valueSize, projSize) };
	nn->addLayer(linformerAttention);
	for (int i = 0; i < numHeads; i++) {
		for (int j = 0; j < keySize; j++) {
			ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
			for (int k = 0; k < *prevSize; k++) {
				linformerAttention->Wq[i].r(j, k) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
		for (int j = 0; j < keySize; j++) {
			ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
			for (int k = 0; k < *prevSize; k++) {
				linformerAttention->Wk[i].r(j, k) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
		for (int j = 0; j < valueSize; j++) {
			ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
			for (int k = 0; k < *prevSize; k++) {
				linformerAttention->Wv[i].r(j, k) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
	}
	for (int i = 0; i < *prevSize - 1; i++) {
		ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < numHeads * valueSize; j++) {
			linformerAttention->Wo.r(i, j) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
		}
	}
	ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
	int maxNumTokens = ModelParser::getNextInt(line, commaIndex, newCommaIndex);
	linformerAttention->E = Matrix(Matrix::ZERO_FILL, projSize, maxNumTokens, true);
	linformerAttention->F = Matrix(Matrix::ZERO_FILL, projSize, maxNumTokens, true);
	for (int i = 0; i < projSize; i++) {
		ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < maxNumTokens; j++) {
			linformerAttention->E.r(i, j) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
		}
	}
	for (int i = 0; i < projSize; i++) {
		ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < maxNumTokens; j++) {
			linformerAttention->F.r(i, j) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
		}
	}
}

void LinformerAttention::setMaxNumTokens(int maxNumTokens) {
	this->maxNumTokens = maxNumTokens;
	float std = 1.0 / maxNumTokens;
	Matrix::NormalFill* normal = new Matrix::NormalFill(0, std);
	if (E.matrix == NULL && F.matrix == NULL) {
		E = Matrix(normal, projSize, maxNumTokens, true);
		F = Matrix(normal, projSize, maxNumTokens, true);
	}
	else if (maxNumTokens > E.maxWidth) {
		throw invalid_argument("Loaded projection matrix is too small");
	}
	if (nextLayer != NULL && instanceOf<Layer2D>(nextLayer)) {
		((Layer2D*)nextLayer)->setMaxNumTokens(maxNumTokens);
	}
}

void LinformerAttention::applyGradients(float learningRate, int t) {
	for (int i = 0; i < batchSize; i++) {
		projOptimizers[0]->addGradient(EGrad[i]);
		projOptimizers[1]->addGradient(FGrad[i]);
		outputOptimizer->addGradient(WoGrad[i]);
		for (int j = 0; j < numHeads; j++) {
			keyOptimizers[j]->addGradient(WkGrad[i][j]);
			queryOptimizers[j]->addGradient(WqGrad[i][j]);
			valueOptimizers[j]->addGradient(WvGrad[i][j]);
		}
	}
	outputOptimizer->applyGradient(Wo, t, learningRate, batchSize);
	projOptimizers[0]->applyGradient(E, t, learningRate, batchSize);
	projOptimizers[1]->applyGradient(F, t, learningRate, batchSize);
	for (int i = 0; i < numHeads; i++) {
		queryOptimizers[i]->applyGradient(Wq[i], t, learningRate, batchSize);
		keyOptimizers[i]->applyGradient(Wk[i], t, learningRate, batchSize);
		valueOptimizers[i]->applyGradient(Wv[i], t, learningRate, batchSize);
	}
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void LinformerAttention::setOptimizer(Optimizer* optimizer) {
	outputOptimizer = optimizer->clone();
	outputOptimizer->setDimensions(size, numHeads * valueSize);
	queryOptimizers = new Optimizer * [numHeads];
	keyOptimizers = new Optimizer * [numHeads];
	valueOptimizers = new Optimizer * [numHeads];
	projOptimizers = new Optimizer * [2] {optimizer->clone(), optimizer->clone()};
	projOptimizers[0]->setDimensions(projSize, maxNumTokens);
	projOptimizers[1]->setDimensions(projSize, maxNumTokens);
	for (int i = 0; i < numHeads; i++) {
		queryOptimizers[i] = optimizer->clone();
		queryOptimizers[i]->setDimensions(keySize, prevSize);
		keyOptimizers[i] = optimizer->clone();
		keyOptimizers[i]->setDimensions(keySize, prevSize);
		valueOptimizers[i] = optimizer->clone();
		valueOptimizers[i]->setDimensions(valueSize, prevSize);
	}
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

int LinformerAttention::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	current += size * numHeads * valueSize;
	current += numHeads * valueSize * prevSize;
	current += 2 * numHeads * keySize * prevSize;
	current += 2 * projSize * maxNumTokens;
	return current;
}