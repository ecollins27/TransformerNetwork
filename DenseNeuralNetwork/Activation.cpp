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

void None::operate(int batchSize, int size, double** activations, double** neurons) {
	return;
}

void None::differentiate(int batchSize, int size, double** activations, double** neurons, double*** activationGradient) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			if (condenseGradient) {
				activationGradient[0][i][j] = 1;
			} else {
				activationGradient[i][j][j] = 1;
			}
		}
	}
}

Activation* None::clone() {
	return { new None() };
}

void Sigmoid::operate(int batchSize, int size, double** activations, double** neurons) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			double& value = neurons[i][j];
			value = 1.0 / (1.0 + exp(-value));
		}
	}
}

void Sigmoid::differentiate(int batchSize, int size, double** activations, double** neurons, double*** activationGradient) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			double value = neurons[i][j];
			if (condenseGradient) {
				activationGradient[0][i][j] = value * (1 - value);
			} else {
				activationGradient[i][j][j] = value * (1 - value);
			}
		}
	}
}

Activation* Sigmoid::clone() {
	return { new Sigmoid() };
}

void Relu::operate(int batchSize, int size, double** activations, double** neurons) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			double& value = neurons[i][j];
			if (value < 0) {
				value = 0;
			}
		}
	}
}

void Relu::differentiate(int batchSize, int size, double** activations, double** neurons, double*** activationGradient) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			double& value = neurons[i][j];
			if (value > 0) {
				if (condenseGradient) {
					activationGradient[0][i][j] = 1;
				} else {
					activationGradient[i][j][j] = 1;
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

void Elu::operate(int batchSize, int size, double** activations, double** neurons) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			double& value = neurons[i][j];
			if (value < 0) {
				value = alpha * (exp(value) - 1);
			}
		}
	}
}

void Elu::differentiate(int batchSize, int size, double** activations, double** neurons, double*** activationGradient) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			double& value = neurons[i][j];
			if (condenseGradient) {
				activationGradient[0][i][j] = value < 0 ? (value + alpha) : 1;
			}
			else {
				activationGradient[i][j][j] = value < 0 ? (value + alpha) : 1;
			}
		}
	}
}

Activation* Elu::clone() {
	return { new Elu(alpha) };
}

void Elu::save(ofstream& file) {
	file << "Elu" << "," << alpha << ",";
}

void Selu::operate(int batchSize, int size, double** activations, double** neurons) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			double& value = neurons[i][j];
			if (value < 0) {
				value = 1.6733 * 1.0507 * (exp(value) - 1);
			}
			else {
				value *= 1.0507;
			}
		}
	}
}

void Selu::differentiate(int batchSize, int size, double** activations, double** neurons, double*** activationGradient) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			double& value = neurons[i][j];
			if (condenseGradient) {
				activationGradient[0][i][j] = value < 0 ? (value + 1.6733 * 1.0507) : 1.0507;
			} else {
				activationGradient[i][j][j] = value < 0 ? (value + 1.6733 * 1.0507) : 1.0507;
			}
		}
	}
}

Activation* Selu::clone() {
	return { new Selu() };
}

void Tanh::operate(int batchSize, int size, double** activations, double** neurons) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			double& value = neurons[i][j];
			double eX = exp(value);
			double eNegX = exp(-value);
			value = (eX - eNegX) / (eX + eNegX);
		}
	}
}

void Tanh::differentiate(int batchSize, int size, double** activations, double** neurons, double*** activationGradient) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			double& value = neurons[i][j];
			if (condenseGradient) {
				activationGradient[0][i][j] = 1 - value * value;
			} else {
				activationGradient[i][j][j] = 1 - value * value;
			}
		}
	}
}

Activation* Tanh::clone() {
	return { new Tanh() };
}

void Swish::operate(int batchSize, int size, double** activations, double** neurons) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			double& value = neurons[i][j];
			value = value / (1 + exp(-value));
		}
	}
}

void Swish::differentiate(int batchSize, int size, double** activations, double** neurons, double*** activationGradient) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			double& activation = activations[i][j];
			double eNegX = exp(-activation);
			if (condenseGradient) {
				activationGradient[0][i][j] = (1 + eNegX * neurons[i][j]) / (1.0 + eNegX);
			} else {
				activationGradient[i][j][j] = (1 + eNegX * neurons[i][j]) / (1.0 + eNegX);
			}
		}
	}
}

Activation* Swish::clone() {
	return { new Swish() };
}

void Softmax::operate(int batchSize, int size, double** activations, double** neurons) {
	for (int i = 0; i < batchSize; i++) {
		int max = 0;
		for (int j = 0; j < size; j++) {
			if (neurons[i][j] > max) {
				max = neurons[i][j];
			}
		}
		double sum = 0;
		for (int j = 0; j < size; j++) {
			double& value = neurons[i][j];
			value = exp(value - max);
			sum += value;
		}
		for (int j = 0; j < size; j++) {
			double& value = neurons[i][j];
			value = value / sum;
		}
	}
}

void Softmax::differentiate(int batchSize, int size, double** activations, double** neurons, double*** activationGradient) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				activationGradient[i][j][k] = neurons[i][j] * ((j == k ? 1 : 0) - neurons[i][k]);
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