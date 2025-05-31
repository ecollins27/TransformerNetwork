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

int main4() {
	cublasCreate(&Matrix2::HANDLE);
	int size = 150;
	int batchSize = 25;
	Matrix* A1 = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, size, size, false);
	Matrix* B1 = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, size, size, true);
	Matrix* C1 = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, size, size, false);

	MatrixBatch A2(batchSize, size, size);
	MatrixBatch B2(batchSize, size, size);
	MatrixBatch C2(batchSize, size, size);

	A2.allocateHost();
	B2.allocateHost();
	C2.allocateHost();
	for (int k = 0; k < batchSize; k++) {
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				A1[k].r(i, j) = (i * size + j) / (float)size;
				B1[k].r(i, j) = (size * size - i * size - j) / (float)size;
				A2(k, i, j) = A1[k](i, j);
				B2(k, i, j) = B1[k](i, j);
				C2(k, i, j) = C1[k](i, j);
			}
		}
	}
	A2.deallocateHost();
	B2.deallocateHost();
	C2.copyToDevice();
	printf("\n%d\n\n", size);
	timeFunction("SIMD", Matrix::multiplyABC, size, size, size, ref(A1[0]), ref(B1[0]), ref(C1[0]), true);
	timeFunction("cuBLAS", MatrixBatch::multiplyABC, ref(A2), ref(B2), ref(C2));

	C2.copyToHost();
	for (int k = 0; k < batchSize; k++) {
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (abs((C1[0](i, j) - C2(k, i, j)) / C1[0](i, j)) > 0.001) {
					printf("Not Equal %f %f %d %d %d\n", C1[0](i, j), C2(k, i, j), k, i, j);
					exit(0);
				}
			}
		}
	}
	printf("Are Equal\n");
}

int main() {
	cublasCreate(&Matrix2::HANDLE);
	int size = 100;
	Matrix A1(Matrix::ZERO_FILL, size, size, false);
	Matrix B1(Matrix::ZERO_FILL, size, size, true);
	Matrix C1(Matrix::ZERO_FILL, size, size, false);

	Matrix2 A2(size, size);
	Matrix2 B2(size, size);
	Matrix2 C2(size, size);

	A2.allocateHost();
	B2.allocateHost();
	C2.allocateHost();
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			A1.r(i, j) = (i * size + j) / (float)size;
			B1.r(i, j) = (size * size - i * size - j) / (float)size;
			A2(i, j) = A1(i, j);
			B2(i, j) = B1(i, j);
			C2(i, j) = C1(i, j);
		}
	}
	A2.deallocateHost();
	B2.deallocateHost();
	C2.copyToDevice();
	printf("\n%d\n\n", size);
	timeFunction("SIMD", Matrix::add, size, size, ref(A1), ref(B1), ref(C1), true);
	timeFunction("cuBLAS", Matrix2::add, ref(A2), ref(B2), ref(C2));

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (abs((C1(i, j) - C2(i, j)) / C1(i, j)) > 0.001) {
				printf("Not Equal\n");
				exit(0);
			}
		}
	}
	printf("Are Equal\n");
}


int main1() {
	int numData = 10000;
	int valData = 1000;
	string* reviews = new string[numData];
	string* valReviews = new string[valData];
	float** y = Matrix::allocateMatrix(Matrix::ZERO_FILL, numData, 2);
	float** yVal = Matrix::allocateMatrix(Matrix::ZERO_FILL, valData, 2);
	getIMDBData("C:\\Users\\Owner\\OneDrive\\Desktop\\IMDB Dataset.csv", reviews, y, 0, numData);
	getIMDBData("C:\\Users\\Owner\\OneDrive\\Desktop\\IMDB Dataset.csv", valReviews, yVal, numData, valData);
	BytePairTokenizer tokenizer("imdb_tokens.txt");

	printf("Size: %d\n", reviews[0].length());
	int* numTokens = new int[numData];
	int* valNumTokens = new int[valData];
	int** X = tokenizer.toSparseTokens(numData, reviews, numTokens);
	int** XVal = tokenizer.toSparseTokens(valData, valReviews, valNumTokens);

	unsigned int maxReviewSize = 0;
	for (int i = 0; i < numData; i++) {
		if (reviews[i].length() > maxReviewSize) {
			maxReviewSize = reviews[i].length();
		}
	}
	printf("MaxReviewSize: %d\n", maxReviewSize);

	Model2DTo1D* model = new Model2DTo1D(1000);
	model->addLayer(new Dense2D(Activation::NONE, 500));
	model->addLayer(new Dense2D(Activation::NONE, 300));
	model->addLayer(new Dense2D(Activation::NONE, 100));
	model->addLayer(new PositionalEncoding2D());
	//model->addTransformerBlock(20, 100, 50);
	//model->addTransformerBlock(20, 100, 30);
	model->addLinformer(20, 100, 10, 10);
	model->addLinformer(20, 100, 10, 10);
	model->addLayer(new Dense2D(Activation::SWISH, 75));
	model->addLayer(new SequenceMean(Activation::NONE));
	model->addLayer(new Dense1D(Activation::SWISH, 20));
	model->addLayer(new Dense1D(Activation::SOFTMAX, 2));

	TrainingParams* params = new TrainingParams(0.00001f, Model2DTo1D::NUM_CORES, 5, 0.1f, Optimizer::ADEMAMIX, new Dataset(valData, valNumTokens, XVal, yVal, true));
	model->fit(new CategoricalCrossEntropy1D(), new Dataset(numData, numTokens, X, y, true), 1, new Loss1D * [1] { new Accuracy1D() }, params);
	model->save("linformer.txt");
	return 0;
}

int main2(int argc, char* args[]) {
	int numData = 10000;
	int valData = 1000;
	string* reviews = new string[numData];
	string* valReviews = new string[valData];
	float** y = Matrix::allocateMatrix(Matrix::ZERO_FILL, numData, 2);
	float** yVal = Matrix::allocateMatrix(Matrix::ZERO_FILL, valData, 2);
	getIMDBData(string(args[1]), reviews, y, 0, numData);
	getIMDBData(string(args[1]), valReviews, yVal, numData, valData);
	string tokenPath(args[2]);
	BytePairTokenizer tokenizer(tokenPath);

	int* numTokens = new int[numData];
	int* valNumTokens = new int[valData];
	int** X = tokenizer.toSparseTokens(numData, reviews, numTokens);
	int** XVal = tokenizer.toSparseTokens(valData, valReviews, valNumTokens);

	unsigned int maxReviewSize = 0;
	for (int i = 0; i < numData; i++) {
		if (reviews[i].length() > maxReviewSize) {
			maxReviewSize = reviews[i].length();
		}
	}
	printf("MaxReviewSize: %d\n", maxReviewSize);

	printf("Constructing Model:\n");
	Model2DTo1D* model = new Model2DTo1D(1000);
	model->addLayer(new Dense2D(Activation::NONE, 500));
	model->addLayer(new Dense2D(Activation::NONE, 300));
	model->addLayer(new Dense2D(Activation::NONE, 100));
	model->addLayer(new PositionalEncoding2D());
	//model->addTransformerBlock(20, 100, 50);
	//model->addTransformerBlock(20, 100, 30);
	model->addLinformer(20, 100, 10, 10);
	model->addLinformer(20, 100, 10, 10);
	model->addLayer(new Dense2D(Activation::SWISH, 75));
	model->addLayer(new SequenceMean(Activation::NONE));
	model->addLayer(new Dense1D(Activation::SWISH, 20));
	model->addLayer(new Dense1D(Activation::SOFTMAX, 2));

	TrainingParams* params = new TrainingParams(0.00001f, Model2DTo1D::NUM_CORES, 5, 0.1f, Optimizer::ADEMAMIX, new Dataset(valData, valNumTokens, XVal, yVal, true));

	printf("Training Model:\n");
	model->fit(new CategoricalCrossEntropy1D(), new Dataset(numData, numTokens, X, y, true), 1, new Loss1D * [1] { new Accuracy1D() }, params);
	model->save("linformer.txt");
	return 0;
}

int main3() {
	int numData = 1000;
	string* reviews = new string[numData];
	float** y = Matrix::allocateMatrix(Matrix::ZERO_FILL, numData, 2);
	getIMDBData("C:\\Users\\Owner\\OneDrive\\Desktop\\IMDB Dataset.csv", reviews, y, 2000, numData);

	BytePairTokenizer tokenizer("imdb_tokens.txt");
	int* numTokens = new int[numData];
	int** X = tokenizer.toSparseTokens(numData, reviews, numTokens);

	Model2DTo1D* model = (Model2DTo1D*)ModelParser::parseModel("linformer.txt");
	model->test(new CategoricalCrossEntropy1D(), new Dataset(numData, numTokens, X, y, true), 1, new Loss1D * [1] { new Accuracy1D() });
	return 0;
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