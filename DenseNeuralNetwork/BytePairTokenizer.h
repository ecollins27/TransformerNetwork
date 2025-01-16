#pragma once
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <sstream>
#include <fstream>
#include "Matrix.h"
using namespace std;


class BytePairTokenizer {

public:
	const static int FREQUENCY_THRESHOLD = 500;
	vector<string> tokenValues;

	BytePairTokenizer(int numStrings, string* strings, int maxTokens);
	BytePairTokenizer(string fileName);
	float*** toTokens(int numStrings, string* strings, int* numTokens);
	void save(string filename);
	float** tokenize(string string, int& length);
};

