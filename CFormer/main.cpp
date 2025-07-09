#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#include "ModelParser.h"
#include "BytePairTokenizer.h"
#include "Matrix2.h"
#include "MatrixBatch.h"
#include <typeinfo>
#include <thread>

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
		X[i] = new float[784];
		y[i] = new float[10];
		for (int k = 0; k < 10; k++) {
			y[i][k] = 0;
		}
		while (getline(ss, n, ',')) {
			int value = stoi(n);
			if (j == 0) {
				y[i][value] = 1;
			}
			else {
				X[i][j - 1] = ((float)value / 255.0) - 0.5;
			}
			j++;
		}
		i++;
	}
	printf("\r100.0");
	printf("\n");
	file.close();
}

void getIMDBData(string fileName, string* X, float** y, int start, int num) {
	string line;
	string sentiment;
	int commaIndex;
	ifstream file(fileName);
	if (file.fail()) {
		throw invalid_argument("Specified file does not exist");
	}
	getline(file, line);
	for (int i = 0; i < start; i++) {
		getline(file, line);
	}
	for (int i = 0; i < num; i++) {
		getline(file, line);
		commaIndex = line.find_last_of(",");
		X[i] = line.substr(0, commaIndex);
		sentiment = line.substr(commaIndex + 1, line.length());
		if (sentiment.compare("positive") == 0) {
			y[i][1] = 1;
			y[i][0] = 0;
		}
		else {
			y[i][1] = 0;
			y[i][0] = 1;
		}
		printf("\r%f", 100.0 * i / num);
	}
	printf("\n");
	file.close();
}

float calculateNaiveAccuracy(int numData, float** y) {
	float mean[2] = { 0, 0 };
	for (int i = 0; i < numData; i++) {
		mean[0] += y[i][0];
		mean[1] += y[i][1];
	}
	mean[0] /= numData;
	mean[1] /= numData;
	if (mean[0] > mean[1]) {
		return mean[0];
	}
	return mean[1];
}

template<typename Function, typename... Params>
long timeFunction(string header, Function function, Params... params) {
	auto start = high_resolution_clock::now();
	function(forward<Params>(params)...);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	printf("%s: %d microseconds\n", header.c_str(), duration.count());
	return duration.count();
}

bool areSimiliar(int m, int p, Matrix& A, Matrix2& B) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			if (abs(((A(i, j) - B(i, j)) / A(i, j))) > 0.001) {
				return false;
			}
		}
	}
	return true;
}

void allocateMatrices(int m, int n, int p, Matrix& A1, Matrix2& A2, Matrix& B1, Matrix2& B2, bool t1, bool t2) {
	if (t1) {
		A1 = Matrix(Matrix::ZERO_FILL, n, m, true);
		A2 = Matrix2(n, m, true);
	}
	else {
		A1 = Matrix(Matrix::ZERO_FILL, m, n, true);
		A2 = Matrix2(m, n, true);
	}
	if (t2) {
		B1 = Matrix(Matrix::ZERO_FILL, p, n, true);
		B2 = Matrix2(p, n, true);
	}
	else {
		B1 = Matrix(Matrix::ZERO_FILL, n, p, true);
		B2 = Matrix2(n, p, true);
	}
	for (int i = 0; i < A2.height; i++) {
		for (int j = 0; j < A2.width; j++) {
			A1.r(i, j) = i * A2.width + j;
			A2(i, j) = A1(i, j);
		}
	}
	for (int i = 0; i < B2.height; i++) {
		for (int j = 0; j < B2.width; j++) {
			B1.r(i, j) = B2.length - i * B2.width - j;
			B2(i, j) = B1(i, j);
		}
	}
	A2.copyToDevice();
	B2.copyToDevice();
}

int main() {
	cublasCreate(&MatrixBatch::HANDLE);

	string filePath = "";
	int numData = 60000;
	float** X = new float* [numData];
	float** y = new float* [numData];
	getMNIST(filePath, X, y, numData);

	Model1D* model = new Model1D(784);
	model->addLayer(new Dense1D(Activation::SWISH, 300));
	model->addLayer(new Dense1D(Activation::SWISH, 100));
	model->addLayer(new Dense1D(Activation::SWISH, 30));
	model->addLayer(new Dense1D(Activation::SOFTMAX, 10));

	TrainingParams* params = TrainingParams::DEFAULT->with<TrainingParams::NUM_EPOCHS>(10);
	model->fit(new CategoricalCrossEntropy1D(), new Dataset(numData, X, y, false), 1, new Loss1D*[1]{new Accuracy1D()}, params);
	model->save("mnist.model");
}

// TODO:
// Finish deconstructors for Layer2D, activations, optimizers, and models
// Implement Performer and Reformer?
// Use SIMD on Normalization and SequenceMean backprop
// Allow Model2D classes to use batch sizes other than NUM_CORES
// 
// 
// Implement RNNs
// Attempt generative model?