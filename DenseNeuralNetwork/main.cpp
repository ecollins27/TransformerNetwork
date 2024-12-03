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

void getIMDBData(string fileName, string* X, float** y, int num) {
	string line;
	int sentiment;
	int commaIndex1, commaIndex2;
	ifstream file(fileName);
	getline(file, line);
	for (int i = 0; i < num; i++) {
		getline(file, line);
		commaIndex1 = line.find_first_of(",");
		commaIndex2 = line.find_first_of(",", commaIndex1 + 1);
		X[i] = line.substr(commaIndex2 + 1, line.length());
		sentiment = stoi(line.substr(commaIndex1 + 1, commaIndex2 - commaIndex1));
		y[i][sentiment] = 1;
		y[i][1 - sentiment] = 0;
		printf("\r%f", 100.0 * i / num);
	}
	printf("\n");
	file.close();
}

int main1() {
	float** X = Matrix::allocateMatrix(Matrix::ZERO_FILL, 60000, 784);
	float** y = Matrix::allocateMatrix(Matrix::ZERO_FILL, 60000, 10);
	getMNIST("C:\\Users\\Owner\\OneDrive\\Desktop\\EMNIST_Data\\emnist-mnist-train.csv", X, y, 60000);
	NeuralNetwork* dnn = { new NeuralNetwork(784) };
	dnn->addLayer({ new DenseLayer(Activation::SWISH, 500) });
	dnn->addLayer({ new DenseLayer(Activation::SWISH, 300) });
	dnn->addLayer({ new DenseLayer(Activation::SWISH, 100) });
	dnn->addLayer({ new LayerNormalization() });
	dnn->addLayer({ new DenseLayer(Activation::SWISH, 50) });
	dnn->addLayer({ new DenseLayer(Activation::SOFTMAX, 10) });
	printf("%d\n", dnn->getNumParameters());
	dnn->fit(Loss::CATEGORICAL_CROSS_ENTROPY, 60000, X, y, 1, &Loss::ACCURACY, TrainingParams::DEFAULT);
	Matrix::deallocateMatrix(X, 60000, 784);
	Matrix::deallocateMatrix(y, 60000, 10);
	return 0;
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

int main() {
	int numData = 100000;
	string* reviews = new string[numData];
	float** y = Matrix::allocateMatrix(Matrix::ZERO_FILL, numData, 2);
	getIMDBData("C:\\Users\\Owner\\OneDrive\\Desktop\\Sentiment Analysis Dataset.csv", reviews, y, numData);
	BytePairTokenizer tokenizer("tokenizer.txt");

	int* numTokens = (int*)malloc(numData * sizeof(int));
	float*** X = tokenizer.toTokens(numData, reviews, numTokens);

	TransformerModel* model{ new TransformerModel(1000) };
	model->addLayer({ new DenseLayer(Activation::NONE, 750) });
	model->addLayer({ new DenseLayer(Activation::NONE, 300) });
	model->addLayer({ new DenseLayer(Activation::NONE, 100) });
	model->addLayer({ new PositionalEncodingLayer() });
//	model->addTransformerBlock(20, 200, 20);
//	model->addTransformerBlock(20, 200, 20);
	model->addTransformerBlock(10, 100, 10);
	model->addTransformerBlock(10, 100, 10);
	model->addLayer({ new DenseLayer(Activation::SWISH, 75) });
	model->addLayer({ new DenseLayer(Activation::SWISH, 20) });
	model->addLayer({ new DenseLayer(Activation::SOFTMAX, 2) });
	model->addLayer({ new BatchMean(Activation::SOFTMAX) });
	
	printf("NumParameters: %d\n", model->getNumParameters());
	printf("NaiveAccuracy: %f\n", calculateNaiveAccuracy(numData, y));
	TrainingParams* params = TrainingParams::DEFAULT->with(TrainingParams::NUM_EPOCHS, 10)->with(TrainingParams::LEARNING_RATE, 0.00001f);
	model->fit(Loss::CATEGORICAL_CROSS_ENTROPY, numData, numTokens, X, y, 1, &Loss::ACCURACY, params, "transformer_model_best.txt");
	model->save("transformer_model.txt");
}

int main2() {
	int numData = 1000;
	string* reviews = new string[numData];
	float** y = Matrix::allocateMatrix(Matrix::ZERO_FILL, numData, 2);
	getIMDBData("C:\\Users\\Owner\\OneDrive\\Desktop\\Sentiment Analysis Dataset.csv", reviews, y, numData);
	BytePairTokenizer tokenizer("tokenizer.txt");

	int* numTokens = (int*)malloc(numData * sizeof(int));
	float*** X = tokenizer.toTokens(numData, reviews, numTokens);

	TransformerModel* model = (TransformerModel*)ModelParser::parseModel("transformer_model_best.txt");
	model->test(Loss::CATEGORICAL_CROSS_ENTROPY, numData, numTokens, X, y, 1, &Loss::ACCURACY);
}