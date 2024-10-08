#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#include "DenseLayer.h"
#include "GatedLayer.h"
#include "NeuralNetwork.h"
#include "BatchNormalization.h"
#include <typeinfo>

using namespace std::chrono;

void getMNIST(string fileName, float** X, float** y, int num) {
	string line;
	ifstream file(fileName);
	int i = 0;
	while (i < num && getline(file, line)) {
		printf("\r%f", 100 * (float)i / (num));
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
				X[i][j - 1] = (float)value / 255.0;
			}
			j++;
		}
		i++;
	}
	printf("\r100.0");
	printf("\n");
	file.close();
}

void getData(string fileName, float** X, float** y, int num) {
	string line;
	ifstream file(fileName);
	getline(file, line);
	int i = 0;
	while (i < num && getline(file, line)) {
		printf("\r%f", 100 * (float)i / (num));
		istringstream ss(line);
		int j = 0;
		string n;
		for (int k = 0; k < 2; k++) {
			y[i][k] = 0;
		}
		while (getline(ss, n, ',')) {
			float value = stod(n);
			if (j == 10) {
				y[i][(int)value] = 1;
			}
			else {
				X[i][j] = (float)value;
			}
			j++;
		}
		i++;
	}
	printf("\r100.0");
	printf("\n");
	file.close();
}

int main2() {
	float** X = Matrix::allocateMatrix(Matrix::ZERO_FILL, 10000, 784);
	float** y = Matrix::allocateMatrix(Matrix::ZERO_FILL, 10000, 10);
	getMNIST("C:\\Users\\Owner\\OneDrive\\Desktop\\EMNIST_Data\\emnist-mnist-test.csv", X, y, 10000);
	NeuralNetwork* dnn{ new NeuralNetwork("dnn.txt")};
	printf("%d\n", dnn->getNumParameters());
	dnn->test(Loss::CATEGORICAL_CROSS_ENTROPY, 10000, X, y, 1, new Loss*[1]{ Loss::ACCURACY });
	Matrix::deallocateMatrix(X, 10000, 784);
	Matrix::deallocateMatrix(y, 10000, 10);
	return 0;
}

int main() {
	int numSamples = 60000;
	int numClasses = 10;
	float** X = Matrix::allocateMatrix(Matrix::ZERO_FILL, numSamples, 784);
	float** y = Matrix::allocateMatrix(Matrix::ZERO_FILL, numSamples, numClasses);
	getMNIST("C:\\Users\\Owner\\OneDrive\\Desktop\\EMNIST_Data\\emnist-mnist-train.csv", X, y, numSamples);
	NeuralNetwork* dnn{ new NeuralNetwork(784) };
	dnn->addLayer({ new BatchNormalization() });
	dnn->addLayer({ new GatedLayer(Activation::SWISH, 1000) });
	dnn->addLayer({ new BatchNormalization() });
	dnn->addLayer({ new GatedLayer(Activation::SWISH, 500) });
	dnn->addLayer({ new BatchNormalization() });
	dnn->addLayer({ new GatedLayer(Activation::SWISH, 100) });
	dnn->addLayer({ new DenseLayer(Activation::SOFTMAX, 10) });
	printf("%d\n", dnn->getNumParameters());
	TrainingParams* params = TrainingParams::DEFAULT->with(TrainingParams::NUM_EPOCHS, 10);
	dnn->fit(Loss::CATEGORICAL_CROSS_ENTROPY, numSamples, X, y, 1, new Loss * [1] {Loss::ACCURACY}, params);
	dnn->save("dnn.txt");
	Matrix::deallocateMatrix(X, numSamples, 784);
	Matrix::deallocateMatrix(y, numSamples, numClasses);
	return 0;
}

/*
* TODO:
* Fix SwiGLU overflowing after 2 layers
* 
* Allow for 2D and 3D layers
* Allow for non sequential models
*/