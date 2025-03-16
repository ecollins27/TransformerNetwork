#pragma once
#include <iostream>
#include <fstream>
#include <random>

using namespace std;


class Dataset {

public:
	int numData;
	int* numTokens;
	void** X;
	void** y;

	bool sparseX;

	template<typename A>
	Dataset(int numData, A** X, float** y, bool sparseX) {
		this->numData = numData;
		this->numTokens = NULL;
		this->X = (void**)X;
		this->y = (void**)y;
		this->sparseX = sparseX;
	}
	template<typename A, typename B>
	Dataset(int numData, int* numTokens, A** X, B** y, bool sparseX) {
		this->numData = numData;
		this->numTokens = numTokens;
		this->X = (void**)X;
		this->y = (void**)y;
		this->sparseX = sparseX;
	}

	Dataset* getMiniBatch(int index, int batchSize);
	int getMaxNumTokens();
	void shuffle();
};

