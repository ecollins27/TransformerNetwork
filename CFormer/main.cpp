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
	if (file.fail()) {
		throw invalid_argument("Specified file does not exist");
	}
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

void getDummyData(int numData, float** X, float** y) {
	NormalFill normalFill(0, 1);
	for (int i = 0; i < numData; i++) {
		X[i] = new float[784];
		y[i] = new float[10];
		for (int j = 0; j < 784; j++) {
			if (j == 0) {
				y[i][j] = 1;
			}
			else if (j < 10) {
				y[i][j] = 0;
			}
			X[i][j] = 0.1;
		}
	}
}

int main() {
	cublasCreate(&MatrixBatch::HANDLE);

	int numData = 60000;
	float** X = new float* [numData];
	float** y = new float* [numData];
	//getDummyData(numData, X, y);
	getMNIST("/mnt/c/Users/eetcollins/Desktop/emnist-mnist-train.csv", X, y, numData);

	Model1D* model = new Model1D(784);
	model->addLayer(new Dense1D(Activation::SWISH, 300));
	model->addLayer(new Dense1D(Activation::SWISH, 100));
	//model->addLayer(new BatchNormalization1D(0.9));
	model->addLayer(new Dense1D(Activation::SWISH, 30));
	model->addLayer(new Dense1D(Activation::SOFTMAX, 10));

	TrainingParams* params = TrainingParams::DEFAULT->with<TrainingParams::NUM_EPOCHS>(10)->with<TrainingParams::OPTIMIZER>(Optimizer::ADEMAMIX)->with<TrainingParams::LEARNING_RATE>(0.001);
	model->fit(new CategoricalCrossEntropy1D(), new Dataset(numData, X, y, false), 1, new Loss1D*[1]{new Accuracy1D()}, params);
	model->save("mnist.model");
	return 0;
}

int main1() {
	cublasCreate(&MatrixBatch::HANDLE);
	int numData = 60000;
	float** X = new float* [numData];
	float** y = new float* [numData];
	//getDummyData(numData, X, y);
	getMNIST("/mnt/c/Users/eetcollins/Desktop/emnist-mnist-train.csv", X, y, numData);

	Model1D* model = (Model1D*)ModelParser::parseModel("mnist.model");
	model->printLayers();
	model->test(new CategoricalCrossEntropy1D(), new Dataset(numData, X, y, false), 1, new Loss1D * [1] {new Accuracy1D()});

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