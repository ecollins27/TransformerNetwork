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
	int num2 = 0;
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
				if (value == 2) {
					num2++;
				}
			}
			else {
				X[i][j - 1] = (double)value / 255.0;
			}
			j++;
		}
		i++;
	}
	printf("\n");
	printf("%d\n", num2);
	file.close();
}

int main() {
	double** X = Matrix::allocateMatrix(Matrix::ZERO_FILL, 60000, 784);
	double** y = Matrix::allocateMatrix(Matrix::ZERO_FILL, 60000, 10);
	getData("C:\\Users\\Owner\\OneDrive\\Desktop\\mnist_train.csv", X, y, 60000);
	NeuralNetwork dnn(Loss::CATEGORICAL_CROSS_ENTROPY, 784);
	dnn.addLayer({ new DenseLayer(Activation::SELU, 500) });
	dnn.addLayer({ new DenseLayer(Activation::SELU, 300) });
	dnn.addLayer({ new DenseLayer(Activation::SELU, 100) });
	dnn.addLayer({ new DenseLayer(Activation::SOFTMAX, 10) });
	dnn.fit(60000, X, y, TrainingParams::DEFAULT->withMetrics(1, Loss::ACCURACY));
}