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

void getIMDBData(string fileName, string* X, float** y, int start, int num) {
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
		y[i][sentiment] = 1;
		y[i][1 - sentiment] = 0;
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

int main() {
	int numData = 100000;
	string* reviews = new string[numData];
	string* valReviews = new string[10000];
	float** y = Matrix::allocateMatrix(Matrix::ZERO_FILL, numData, 2);
	float** yVal = Matrix::allocateMatrix(Matrix::ZERO_FILL, 10000, 2);
	getIMDBData("C:\\Users\\Owner\\OneDrive\\Desktop\\Sentiment Analysis Dataset.csv", reviews, y, 0, numData);
	getIMDBData("C:\\Users\\Owner\\OneDrive\\Desktop\\Sentiment Analysis Dataset.csv", valReviews, yVal, numData, 10000);
	BytePairTokenizer tokenizer("tokens1000.txt");

	int* numTokens = new int[numData];
	int* valNumTokens = new int[10000];
	float*** X = tokenizer.toTokens(numData, reviews, numTokens);
	float*** XVal = tokenizer.toTokens(10000, valReviews, valNumTokens);

	Model2DTo1D* model{ new Model2DTo1D(1000) };
	model->addLayer({ new Dense2D(Activation::NONE, 750) });
	model->addLayer({ new Dense2D(Activation::NONE, 300) });
	model->addLayer({ new Dense2D(Activation::NONE, 100) });
	model->addLayer({ new PositionalEncoding2D() });
	model->addTransformerBlock(20, 100, 50);
	model->addTransformerBlock(20, 100, 30);
	model->addTransformerBlock(20, 100, 20);
	model->addTransformerBlock(20, 100, 10);
	model->addLayer({ new Dense2D(Activation::SWISH, 75) });
	model->addLayer({ new Dense2D(Activation::SWISH, 20) });
	model->addLayer({ new Dense2D(Activation::SWISH, 2) });
	model->addLayer({ new SequenceMean(Activation::SOFTMAX) });
	
	printf("NumParameters: %d\n", model->getNumParameters());
	printf("NaiveAccuracy: %f\n", calculateNaiveAccuracy(numData, y));
	Optimizer* optimizer = new AdEMAMix(0.9, 0.9999, 0.999, 5, 0.00001);
	TrainingParams* params = new TrainingParams(0.00001f, 12, 20, 0.1f, Optimizer::ADEMAMIX, 10000, valNumTokens, XVal, yVal);
	model->fit(new CategoricalCrossEntropy1D(), numData, numTokens, X, y, 1, new Loss1D*[1]{ new Accuracy1D() }, params);
	model->save("transformer4_regular.txt");
	return 0;
}

int main2() {
	int numData = 10000;
	string* reviews = new string[numData];
	float** y = Matrix::allocateMatrix(Matrix::ZERO_FILL, numData, 2);
	getIMDBData("C:\\Users\\Owner\\OneDrive\\Desktop\\Sentiment Analysis Dataset.csv", reviews, y, 100000, numData);
	BytePairTokenizer tokenizer("tokens1000.txt");

	int* numTokens = new int[numData];
	float*** X = tokenizer.toTokens(numData, reviews, numTokens);

	Model2DTo1D* model = (Model2DTo1D*)ModelParser::parseModel("transformer4_regular.txt");

	model->test(new CategoricalCrossEntropy1D(), numData, numTokens, X, y, 1, new Loss1D * [1] { new Accuracy1D()});
}

int main3() {
	int numData = 200000;
	string* reviews = new string[numData];
	float** y = Matrix::allocateMatrix(Matrix::ZERO_FILL, numData, 2);
	getIMDBData("C:\\Users\\Owner\\OneDrive\\Desktop\\Sentiment Analysis Dataset.csv", reviews, y, 0, numData);

	BytePairTokenizer tokenizer(numData, reviews, 1000);
	tokenizer.save("tokens1000.txt");
}

// TODO:
// Use SIMD on Normalization layers
// Allow Model2D classes to use batch sizes other than NUM_CORES
// Store __m128 objects permanently in matrix class?
// 
// 
// Implement RNNs
// Attempt generative model?