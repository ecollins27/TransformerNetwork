#include "Activation.h"
#include "DenseLayer.h"

Activation* Activation::NONE{new None()};
Activation* Activation::SIGMOID{ new Sigmoid() };
Activation* Activation::RELU{ new Relu() };
Activation* Activation::ELU{ new Elu(1) };
Activation* Activation::SELU{ new Selu() };
Activation* Activation::TANH{ new Tanh() };
Activation* Activation::SOFTMAX{ new Softmax() };
Activation* Activation::ALL_ACTIVATIONS[7] = { NONE, SIGMOID, RELU, ELU, SELU, TANH, SOFTMAX };

void None::operate(DenseLayer* layer) {
	return;
}

void None::differentiate(DenseLayer* layer) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			layer->activationGradient[0][i][j] = 1;
		}
	}
}

bool None::isDiagonal() {
	return true;
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
			layer->activationGradient[0][i][j] = value * (1 - value);
		}
	}
}

bool Sigmoid::isDiagonal() {
	return true;
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
				layer->activationGradient[0][i][j] = 1;
			}
		}
	}
}

bool Relu::isDiagonal() {
	return true;
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
			layer->activationGradient[0][i][j] = value < 0 ? (value + alpha) : 1;
		}
	}
}

bool Elu::isDiagonal() {
	return true;
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
			layer->activationGradient[0][i][j] = value < 0 ? (value + 1.6733 * 1.0507) : 1.0507;
		}
	}
}

bool Selu::isDiagonal() {
	return true;
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
			layer->activationGradient[0][i][j] = 1 - value * value;
		}
	}
}

bool Tanh::isDiagonal() {
	return true;
}

void Softmax::operate(DenseLayer* layer) {
	for (int i = 0; i < layer->batchSize; i++) {
		double sum = 0;
		for (int j = 0; j < layer->size; j++) {
			double& value = layer->neurons[i][j];
			double expValue = exp(value);
			value = exp(value);
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

bool Softmax::isDiagonal() {
	return false;
}