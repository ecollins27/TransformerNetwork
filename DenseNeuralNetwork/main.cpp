#include <iostream>
#include <fstream>
#include <sstream>

#include "DenseLayer.h"
#include "GAN.h"

void getData(string fileName, double** X, double** y, int num) {
	string line;
	ifstream file(fileName);
	getline(file, line);
	int i = 0;
	while (i < num && getline(file, line)) {
		printf("\r%f", 100 * (double)i / (num));
		istringstream ss(line);
		int j = 0;
		string n;
		for (int k = 0; k < 10; k++) {
			y[i][k] = 0;
		}
		while (getline(ss, n, ',')) {
			int value = stoi(n);
			if (j == 0) {
				y[i][value] = 1;
			}
			else {
				X[i][j - 1] = (double)value / 255.0;
			}
			j++;
		}
		i++;
	}
	printf("\r100.0 \n");
	file.close();
}

int main() {
	double** X = Matrix::allocateMatrix(Matrix::ZERO_FILL, 10000, 2);
	double** y = Matrix::allocateMatrix(Matrix::ZERO_FILL, 10000, 2);
	for (int i = 0; i < 10000; i++) {
		X[i][0] = (int)(2.0 * rand() / (RAND_MAX + 1));
		X[i][1] = (int)(2.0 * rand() / (RAND_MAX + 1));
		y[i][(int)(X[i][0] + X[i][1]) % 2] = 1;
	}
	NeuralNetwork dnn(Loss::BINARY_CROSS_ENTROPY, 2);
	dnn.addLayer({ new DenseLayer(Activation::SIGMOID, 2) });
	dnn.addLayer({ new DenseLayer(Activation::SOFTMAX, 2) });
	dnn.fit(10000, X, y, TrainingParams::DEFAULT->withMetrics(1, Loss::ACCURACY));
	printf("\n");
	double** weight1 = ((DenseLayer*)(dnn.inputLayer->nextLayer))->weights;
	double** weight2 = ((DenseLayer*)(dnn.outputLayer))->weights;
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%f ", weight1[i][j]);
		}
		printf("\n");
	}
	printf("\n");
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%f ", weight2[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

int main2() {
	double** X = Matrix::allocateMatrix(Matrix::ZERO_FILL, 60000, 784);
	double** y = Matrix::allocateMatrix(Matrix::ZERO_FILL, 60000, 10);
	getData("C:\\Users\\Owner\\OneDrive\\Desktop\\mnist_train.csv", X, y, 60000);
	NeuralNetwork* generator{ new NeuralNetwork(Loss::MEAN_SQUARED_ERROR, 10) };
	generator->addLayer({ new DenseLayer(Activation::SELU, 300) });
	generator->addLayer({ new DenseLayer(Activation::SIGMOID, 784) });
	NeuralNetwork* discriminator{ new NeuralNetwork(Loss::BINARY_CROSS_ENTROPY, 784) };
	discriminator->addLayer({ new DenseLayer(Activation::SELU, 300) });
	discriminator->addLayer({ new DenseLayer(Activation::SIGMOID, 1) });
	GAN gan(generator, discriminator);
	gan.fit(60000, X, TrainingParams::DEFAULT);
	return 0;
}