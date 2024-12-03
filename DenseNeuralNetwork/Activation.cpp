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

void None::operate(int batchSize, int size, float** activations, float** neurons) {
	Matrix::copy(batchSize, size, activations, neurons);
	return;
}

void None::differentiate(int batchSize, int size, float** activations, float** neurons, float*** activationGradient) {
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

void Sigmoid::operate(int batchSize, int size, float** activations, float** neurons) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			float value = activations[i][j];
			neurons[i][j] = 1.0 / (1.0 + exp(-value));
		}
	}
}

void Sigmoid::differentiate(int batchSize, int size, float** activations, float** neurons, float*** activationGradient) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			float value = neurons[i][j];
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

void Relu::operate(int batchSize, int size, float** activations, float** neurons) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			float value = activations[i][j];
			if (value < 0) {
				neurons[i][j] = 0;
			}
			else {
				neurons[i][j] = value;
			}
		}
	}
}

void Relu::differentiate(int batchSize, int size, float** activations, float** neurons, float*** activationGradient) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			float& value = neurons[i][j];
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

Elu::Elu(float alpha) {
	this->alpha = alpha;
}

void Elu::operate(int batchSize, int size, float** activations, float** neurons) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			float value = activations[i][j];
			if (value < 0) {
				neurons[i][j] = alpha * (exp(value) - 1);
			}
			else {
				neurons[i][j] = value;
			}
		}
	}
}

void Elu::differentiate(int batchSize, int size, float** activations, float** neurons, float*** activationGradient) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			float& value = neurons[i][j];
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

void Selu::operate(int batchSize, int size, float** activations, float** neurons) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			float value = neurons[i][j];
			if (value < 0) {
				neurons[i][j] = 1.6733 * 1.0507 * (exp(value) - 1);
			}
			else {
				neurons[i][j] = value * 1.0507;
			}
		}
	}
}

void Selu::differentiate(int batchSize, int size, float** activations, float** neurons, float*** activationGradient) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			float& value = neurons[i][j];
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

void Tanh::operate(int batchSize, int size, float** activations, float** neurons) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			float value = neurons[i][j];
			float eX = exp(value);
			float eNegX = exp(-value);
			neurons[i][j] = (eX - eNegX) / (eX + eNegX);
		}
	}
}

void Tanh::differentiate(int batchSize, int size, float** activations, float** neurons, float*** activationGradient) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			float& value = neurons[i][j];
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

void Swish::operate(int batchSize, int size, float** activations, float** neurons) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			float value = activations[i][j];
			neurons[i][j] = value / (1 + exp(-value));
		}
	}
}

void Swish::differentiate(int batchSize, int size, float** activations, float** neurons, float*** activationGradient) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			if (condenseGradient) {
				activationGradient[0][i][j] = activations[i][j] == 0? 0.5:(neurons[i][j] * (activations[i][j] - neurons[i][j] + 1) / activations[i][j]);
			} else {
				activationGradient[i][j][j] = activations[i][j] == 0 ? 0.5 : (neurons[i][j] * (activations[i][j] - neurons[i][j] + 1) / activations[i][j]);
			}
		}
	}
}

Activation* Swish::clone() {
	return { new Swish() };
}

void Softmax::operate(int batchSize, int size, float** activations, float** neurons) {
	for (int i = 0; i < batchSize; i++) {
		int max = INT_MIN;
		for (int j = 0; j < size; j++) {
			if (activations[i][j] > max) {
				max = activations[i][j];
			}
		}
		float sum = 0;
		for (int j = 0; j < size; j++) {
			neurons[i][j] = exp(activations[i][j] - max + 16);
			sum += neurons[i][j];
		}
		for (int j = 0; j < size; j++) {
			neurons[i][j] /= sum;
		}
	}
}

void Softmax::differentiate(int batchSize, int size, float** activations, float** neurons, float*** activationGradient) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				activationGradient[i][k][j] = neurons[i][j] * ((j == k ? 1 : 0) - neurons[i][k]);
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