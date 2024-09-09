#include "GAN.h"


GAN::GAN(NeuralNetwork* generator, NeuralNetwork* discriminator) {
	this->generator = generator;
	this->discriminator = discriminator;
}

void GAN::setCoding() {
	for (int i = 0; i < generator->inputLayer->size; i++) {
		generator->inputLayer->neurons[i][0] = (double)rand() / (RAND_MAX + 1);
	}
}

void GAN::forwardPropagate() {
	setCoding();
	generator->inputLayer->forwardPropagate();
	for (int i = 0; i < generator->outputLayer->size; i++) {
		discriminator->inputLayer->neurons[i][0] = generator->outputLayer->neurons[i][0];
	}
	discriminator->inputLayer->forwardPropagate();
}

void GAN::backPropagate(double* yTrue) {
	discriminator->backPropagate(yTrue);
	for (int i = 0; i < generator->outputLayer->size; i++) {
		generator->outputLayer->neuronGradient[i][0] = discriminator->inputLayer->neuronGradient[i][0];
	}
	generator->outputLayer->backPropagate();
}

void GAN::fit(double* X, double* losses, TrainingParams* params) {
	discriminator->setTrainable(true);
	discriminator->forwardPropagate(X);
	discriminator->backPropagate(one);

	forwardPropagate();
	generator->setTrainable(false);
	discriminator->backPropagate(zero);

	for (int i = 0; i < params->numMetrics; i++) {
		losses[i] += params->metrics[i]->loss(discriminator->outputLayer, one);
	}
	losses[params->numMetrics] += discriminator->lossFunction->loss(discriminator->outputLayer, one);
	generator->setTrainable(true);
	discriminator->setTrainable(false);
	backPropagate(one);
}

void GAN::writeToFile() {
	setCoding();
	generator->inputLayer->forwardPropagate();
	string fileName = "output.ppm";
	ofstream file(fileName.c_str());
	file << "P3 28 28 256";
	for (int i = 0; i < 784; i++) {
		if (i % 28 == 0) {
			file << "\n";
		}
		int value = (int)(generator->outputLayer->neurons[i][0] * 256);
		for (int j = 0; j < 3; j++) {
			file << value << " ";
		}
	}
	file.close();
}

void GAN::shuffle(int numData, double** X) {
	for (int i = 0; i < numData; i++) {
		int index = (int)(numData * ((double)rand() / (RAND_MAX + 1)));
		swap(X[i], X[index]);
	}
}

void GAN::fit(int numData, double** X, TrainingParams* params) {
	double* averages = new double[params->numMetrics + 1];
	for (int epoch = 0; epoch < params->numEpochs; epoch++) {
		shuffle(numData, X);
		for (int i = 0; i < params->numMetrics + 1; i++) {
			averages[i] = 0;
		}
		for (int i = 0; i < numData; i++) {
			if (i % (params->batchSize / 2) == 0) {
				discriminator->applyGradients(params);
				printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, params->numEpochs, i + 1, numData, averages[params->numMetrics] / (2 * i + 2));
				for (int j = 0; j < params->numMetrics; j++) {
					printf("%s:%f  ", params->metrics[j]->toString().c_str(), averages[j] / (2 * i + 2));
				}
			} else if (i % params->batchSize == 0) {
				generator->applyGradients(params);
			}
			fit(X[i], averages, params);
		}
		discriminator->applyGradients(params);
		generator->applyGradients(params);
		printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, params->numEpochs, numData, numData, averages[params->numMetrics] / (2 * numData));
		for (int j = 0; j < params->numMetrics; j++) {
			printf("%s:%f  ", params->metrics[j]->toString().c_str(), averages[j] / (2 * numData));
		}
		for (int i = 0; i < params->numMetrics + 1; i++) {
			averages[i] = 0;
		}
		writeToFile();
		printf("\n");
	}
}