#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#include "ModelParser.h"
#include "BytePairTokenizer.h"
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

void getIMDBData(string fileName, string* X, float*** y, int start, int num) {
	string line;
	int sentiment;
	int commaIndex1, commaIndex2;
	ifstream file(fileName);
	getline(file, line);
	for (int i = 0; i < start; i++) {
		getline(file, line);
	}
	for (int i = 0; i < num; i++) {
		getline(file, line);
		commaIndex1 = line.find_first_of(",");
		commaIndex2 = line.find_first_of(",", commaIndex1 + 1);
		X[i] = line.substr(commaIndex2 + 1, line.length());
		sentiment = stoi(line.substr(commaIndex1 + 1, commaIndex2 - commaIndex1));
		y[0][i][sentiment] = 1;
		y[0][i][1 - sentiment] = 0;
		printf("\r%f", 100.0 * i / num);
	}
	printf("\n");
	file.close();
}

float calculateNaiveAccuracy(int numData, float*** y) {
	float mean[2] = { 0, 0 };
	for (int i = 0; i < numData; i++) {
		mean[0] += y[0][i][0];
		mean[1] += y[0][i][1];
	}
	mean[0] /= numData;
	mean[1] /= numData;
	if (mean[0] > mean[1]) {
		return mean[0];
	}
	return mean[1];
}

int main() {
	int numData = 100000;
	string* reviews = new string[numData];
	float*** y = Matrix::allocateMatrix3D(Matrix::ZERO_FILL, 1, numData, 2);
	getIMDBData("C:\\Users\\Owner\\OneDrive\\Desktop\\Sentiment Analysis Dataset.csv", reviews, y, 0, numData);
	BytePairTokenizer tokenizer("tokenizer.txt");

	int* numTokens = (int*)malloc(numData * sizeof(int));
	float*** X = tokenizer.toTokens(numData, reviews, numTokens);

	Model2D* model{ new Model2D(1000) };
	model->addLayer({ new Dense2D(Activation::NONE, 750) });
	model->addLayer({ new Dense2D(Activation::NONE, 300) });
	model->addLayer({ new Dense2D(Activation::NONE, 100) });
	model->addLayer({ new PositionalEncoding2D() });
	model->addTransformerBlock(10, 100, 50);
	model->addTransformerBlock(10, 100, 50);
	model->addTransformerBlock(10, 100, 50);
	model->addLayer({ new Dense2D(Activation::SWISH, 75) });
	model->addLayer({ new Dense2D(Activation::SWISH, 20) });
	model->addLayer({ new Dense2D(Activation::SWISH, 2) });
	model->addLayer({ new SequenceMean(Activation::SOFTMAX) });
	
	printf("NumParameters: %d\n", model->getNumParameters());
	printf("NaiveAccuracy: %f\n", calculateNaiveAccuracy(numData, y));
	TrainingParams* params = TrainingParams::DEFAULT->with(TrainingParams::NUM_EPOCHS, 10)->with(TrainingParams::LEARNING_RATE, 0.00001f);
	Optimizer* optimizer = { new AdEMAMix(0.9, 0.9999, 0.999, 5, 0.0001) };
	params = params->with(TrainingParams::OPTIMIZER, optimizer)->with(TrainingParams::BATCH_SIZE, 12);
	model->oneThreadFit(new CategoricalCrossEntropy1D(), numData, numTokens, X, y, 1, new Loss*[1]{ new Accuracy1D() }, params, "transformer4_regular.txt");
	return 0;
}

int main1() {
	float** X = Matrix::allocateMatrix(Matrix::ZERO_FILL, 60000, 784);
	float** y = Matrix::allocateMatrix(Matrix::ZERO_FILL, 60000, 10);
	getMNIST("C:\\Users\\Owner\\OneDrive\\Desktop\\EMNIST_Data\\emnist-mnist-train.csv", X, y, 60000);

	Model1D* model{ new Model1D(784) };
	model->addLayer(new Gated1D(Activation::SWISH, 500));
	model->addLayer(new Gated1D(Activation::SWISH, 300));
	model->addLayer(new Dropout1D(0.5));
	model->addLayer(new Gated1D(Activation::SWISH, 100));
	model->addLayer(new Dense1D(Activation::SOFTMAX, 10));
	TrainingParams* params = TrainingParams::DEFAULT->with(TrainingParams::NUM_EPOCHS, 20);
	model->fit(new CategoricalCrossEntropy1D(), 60000, X, y, 1, new Loss * [1] {new Accuracy1D()}, params);
}

// TODO:
// Optimize as much as possible - consider using better hardware acceleration and/or actual matrix library
// Use mini-batch gradient descent instead of stochastic
// 
// 
// Implement RNNs
// Attempt generative model?