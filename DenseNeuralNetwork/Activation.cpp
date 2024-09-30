#include "Activation.h"
#include "DenseLayer.h"

Activation* Activation::NONE{new None()};
Activation* Activation::SIGMOID{ new Sigmoid() };
Activation* Activation::RELU{ new Relu() };
Activation* Activation::ELU{ new Elu(1) };
Activation* Activation::SELU{ new Selu() };
Activation* Activation::TANH{ new Tanh() };
Activation* Activation::SWISH{ new Swish() };
Activation* Activation::SOFTMAX{ new Softmax() };
Activation* Activation::ALL_ACTIVATIONS[Activation::NUM_ACTIVATIONS] = {NONE, SIGMOID, RELU, ELU, SELU, TANH, SWISH, SOFTMAX };

void None::operate(DenseLayer* layer) {
	return;
}

void None::differentiate(DenseLayer* layer) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			if (condenseGradient) {
				layer->activationGradient[0][i][j] = 1;
			} else {
				layer->activationGradient[i][j][j] = 1;
			}
		}
	}
}

Activation* None::clone() {
	return { new None() };
}

void Sigmoid::operate(DenseLayer* layer) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			double& value = layer->neurons[i][j];
			value = 1.0 / (1.0 + exp(-value));
		}
	}
}

void Sigmoid::differentiate(DenseLayer* layer) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			double value = layer->neurons[i][j];
			if (condenseGradient) {
				layer->activationGradient[0][i][j] = value * (1 - value);
			} else {
				layer->activationGradient[i][j][j] = value * (1 - value);
			}
		}
	}
}

Activation* Sigmoid::clone() {
	return { new Sigmoid() };
}

void Relu::operate(DenseLayer* layer) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			double& value = layer->neurons[i][j];
			if (value < 0) {
				value = 0;
			}
		}
	}
}

void Relu::differentiate(DenseLayer* layer) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			double& value = layer->neurons[i][j];
			if (value > 0) {
				if (condenseGradient) {
					layer->activationGradient[0][i][j] = 1;
				} else {
					layer->activationGradient[i][j][j] = 1;
				}
			}
		}
	}
}

Activation* Relu::clone() {
	return { new Relu() };
}

Elu::Elu(double alpha) {
	this->alpha = alpha;
}

void Elu::operate(DenseLayer* layer) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			double& value = layer->neurons[i][j];
			if (value < 0) {
				value = alpha * (exp(value) - 1);
			}
		}
	}
}

void Elu::differentiate(DenseLayer* layer) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			double& value = layer->neurons[i][j];
			if (condenseGradient) {
				layer->activationGradient[0][i][j] = value < 0 ? (value + alpha) : 1;
			}
			else {
				layer->activationGradient[i][j][j] = value < 0 ? (value + alpha) : 1;
			}
		}
	}
}

Activation* Elu::clone() {
	return { new Elu(alpha) };
}

void Selu::operate(DenseLayer* layer) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			double& value = layer->neurons[i][j];
			if (value < 0) {
				value = 1.6733 * 1.0507 * (exp(value) - 1);
			}
			else {
				value *= 1.0507;
			}
		}
	}
}

void Selu::differentiate(DenseLayer* layer) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			double& value = layer->neurons[i][j];
			if (condenseGradient) {
				layer->activationGradient[0][i][j] = value < 0 ? (value + 1.6733 * 1.0507) : 1.0507;
			} else {
				layer->activationGradient[i][j][j] = value < 0 ? (value + 1.6733 * 1.0507) : 1.0507;
			}
		}
	}
}

Activation* Selu::clone() {
	return { new Selu() };
}

void Tanh::operate(DenseLayer* layer) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			double& value = layer->neurons[i][j];
			double eX = exp(value);
			double eNegX = exp(-value);
			value = (eX - eNegX) / (eX + eNegX);
		}
	}
}

void Tanh::differentiate(DenseLayer* layer) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			double& value = layer->neurons[i][j];
			if (condenseGradient) {
				layer->activationGradient[0][i][j] = 1 - value * value;
			} else {
				layer->activationGradient[i][j][j] = 1 - value * value;
			}
		}
	}
}

Activation* Tanh::clone() {
	return { new Tanh() };
}

void Swish::operate(DenseLayer* layer) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			double& value = layer->neurons[i][j];
			value = value / (1 + exp(-value));
		}
	}
}

void Swish::differentiate(DenseLayer* layer) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			double& activation = layer->activations[i][j];
			double eNegX = exp(-activation);
			if (condenseGradient) {
				layer->activationGradient[0][i][j] = (1 + eNegX * layer->neurons[i][j]) / (1.0 + eNegX);
			} else {
				layer->activationGradient[i][j][j] = (1 + eNegX * layer->neurons[i][j]) / (1.0 + eNegX);
			}
		}
	}
}

Activation* Swish::clone() {
	return { new Swish() };
}

Glu::Glu(Activation* activation) {
	this->activation = activation->clone();
	this->activation->condenseGradient = false;
}

void Glu::operate(DenseLayer* layer) {
	activation->operate(layer);
	Matrix::copy(layer->batchSize, layer->size, layer->neurons, activationOutput);
	Matrix::matrixMultiplyABtC(layer->batchSize, layer->size + 1, layer->size, layer->activations, weights, output, true);
	Matrix::elementMultiply(layer->batchSize, layer->size, activationOutput, output, layer->neurons, true);
}

void Glu::differentiate(DenseLayer* layer) {
	for (int i = 0; i < layer->batchSize; i++) {
		Matrix::fill(Matrix::ZERO_FILL, layer->size, layer->size, layer->activationGradient[i]);
	}
	activation->differentiate(layer);
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			for (int k = 0; k < layer->size; k++) {
				layer->activationGradient[i][j][k] = layer->activationGradient[i][j][k] * output[i][j] + activationOutput[i][j] * weights[j][k];
				weightGradient[j][k] += layer->neuronGradient[i][j] * activationOutput[i][j] * layer->activations[i][k];
			}
		}
	}
}

Activation* Glu::clone() {
	return { new Glu(activation) };
}

template<typename A, typename T>
bool instanceof(T* ptr) {
	return dynamic_cast<A*>(ptr) != NULL;
}

void Glu::init(DenseLayer* layer) {
	if (weights == NULL) {
		double stdDeviation = sqrt(2.0 / (layer->size));
		if (instanceof<Selu>(activation)) {
			stdDeviation = sqrt(1.0 / layer->size);
		}
		weights = Matrix::allocateMatrix({ new Matrix::NormalFill(0,stdDeviation) }, layer->size, layer->size + 1);
		weightGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, layer->size, layer->size + 1);
	}
	output = Matrix::allocateMatrix(Matrix::ZERO_FILL, layer->batchSize, layer->size);
	activationOutput = Matrix::allocateMatrix(Matrix::ZERO_FILL, layer->batchSize, layer->size);
}

void Glu::setOptimizer(DenseLayer* layer, Optimizer* optimizer) {
	this->optimizer = optimizer->clone();
	this->optimizer->setDimensions(layer->size, layer->size + 1);
}

void Glu::applyGradient(DenseLayer* layer, TrainingParams* params, int t) {
	Matrix::scale(layer->size, layer->size + 1, weightGradient, 1.0 / layer->batchSize);
	optimizer->applyGradient(weightGradient, weights, t, params);
}

bool Glu::isDiagonal() {
	return false;
}

void Softmax::operate(DenseLayer* layer) {
	for (int i = 0; i < layer->batchSize; i++) {
		int max = 0;
		for (int j = 0; j < layer->size; j++) {
			if (layer->neurons[i][j] > max) {
				max = layer->neurons[i][j];
			}
		}
		double sum = 0;
		for (int j = 0; j < layer->size; j++) {
			double& value = layer->neurons[i][j];
			value = exp(value - max);
			sum += value;
		}
		for (int j = 0; j < layer->size; j++) {
			double& value = layer->neurons[i][j];
			value = value / sum;
		}
	}
}

void Softmax::differentiate(DenseLayer* layer) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			for (int k = 0; k < layer->size; k++) {
				layer->activationGradient[i][j][k] = layer->neurons[i][j] * ((j == k ? 1 : 0) - layer->neurons[i][k]);
			}
		}
	}
}

Activation* Softmax::clone() {
	return { new Softmax() };
}

bool Softmax::isDiagonal() {
	return false;
}