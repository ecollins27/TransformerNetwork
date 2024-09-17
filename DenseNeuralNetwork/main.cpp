#include <iostream>
#include <fstream>
#include <sstream>

#include "DenseLayer.h"
#include "NeuralNetwork.h"

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
	printf("\r100.0");
	printf("\n");
	file.close();
}

int main2() {
	double** X = Matrix::allocateMatrix(Matrix::ZERO_FILL, 10000, 784);
	double** y = Matrix::allocateMatrix(Matrix::ZERO_FILL, 10000, 10);
	getData("C:\\Users\\Owner\\OneDrive\\Desktop\\mnist_test.csv", X, y, 10000);
	NeuralNetwork* dnn{ new NeuralNetwork("dnn.txt")};
	dnn->test(10000, X, y, 1, new Loss*[1]{ Loss::ACCURACY });

	Matrix::deallocateMatrix(X, 60000, 784);
	Matrix::deallocateMatrix(y, 60000, 10);
}

int main() {
	double** X = Matrix::allocateMatrix(Matrix::ZERO_FILL, 60000, 784);
	double** y = Matrix::allocateMatrix(Matrix::ZERO_FILL, 60000, 10);
	getData("C:\\Users\\Owner\\OneDrive\\Desktop\\mnist_train.csv", X, y, 60000);
	NeuralNetwork* dnn{ new NeuralNetwork(Loss::CATEGORICAL_CROSS_ENTROPY, 784) };
	dnn->addLayer({ new DenseLayer(Activation::SWISH, 500) });
	dnn->addLayer({ new DenseLayer(Activation::SWISH, 300) });
	dnn->addLayer({ new DenseLayer(Activation::SWISH, 100) });
	dnn->addLayer({ new DenseLayer(Activation::SOFTMAX, 10) });
	dnn->fit(60000, X, y, TrainingParams::DEFAULT->withMetrics(1, Loss::ACCURACY)->withNumEpochs(30));
	dnn->save("dnn.txt");

	Matrix::deallocateMatrix(X, 60000, 784);
	Matrix::deallocateMatrix(y, 60000, 10);
	return 0;
}