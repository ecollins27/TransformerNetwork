#include "BytePairTokenizer.h"

bool contains(vector<string> strings, string str) {
	for (string s : strings) {
		if (s.compare(str) == 0) {
			return true;
		}
	}
	return false;
}

void setNextToken(vector<string> tokens, int** frequencyMap, int maxIndex[2]) {
	int maxFrequency = BytePairTokenizer::FREQUENCY_THRESHOLD - 1;
	for (int i = 0; i < tokens.size(); i++) {
		for (int j = 0; j < tokens.size(); j++) {
			if (frequencyMap[i][j] > maxFrequency) {
				maxFrequency = frequencyMap[i][j];
				maxIndex[0] = i;
				maxIndex[1] = j;
			}
		}
	}
}

int getIndex(char c, vector<string> tokens) {
	for (int i = 0; i < tokens.size(); i++) {
		if (tokens[i][0] == c) {
			return i;
		}
	}
	return -1;
}

void fillFrequencyMap(int numStrings, string* strings, vector<string> tokens, int** frequencyMap) {
	int index1, index2, index3;
	for (int i = 0; i < numStrings; i++) {
		index1 = 0;
		index2 = strings[i].find_first_of(',', 1);
		index3 = strings[i].find_first_of(',', index2 + 1);
		while (index3 < strings[i].length() - 1) {
			int row = stoi(strings[i].substr(index1 + 1, index2 - index1 - 1));
			int column = stoi(strings[i].substr(index2 + 1, index3 - index2 - 1));
			frequencyMap[row][column]++;
			index1 = index2;
			index2 = index3;
			index3 = strings[i].find_first_of(',', index3 + 1);
		}
	}
}

void replaceStrings(int numStrings, string* strings, int maxIndex[2], int tokenNum) {
	string toReplace = "," + to_string(maxIndex[0]) + "," + to_string(maxIndex[1]) + ",";
	int len = toReplace.length();
	string replaceWith = "," + to_string(tokenNum) + ",";
	for (int i = 0; i < numStrings; i++) {
		int index = strings[i].find(toReplace);
		while (index != string::npos) {
			strings[i].replace(index, len, replaceWith);
			index = strings[i].find(toReplace);
		}
	}
}

int calculateNextToken(int numStrings, string* newStrings, vector<string> tokenValues, int maxIndex[2]) {
	int** frequencyMap = (int**)malloc(tokenValues.size() * sizeof(int*));
	for (int i = 0; i < tokenValues.size(); i++) {
		frequencyMap[i] = (int*)malloc(tokenValues.size() * sizeof(int));
		for (int j = 0; j < tokenValues.size(); j++) {
			frequencyMap[i][j] = 0;
		}
	}
	maxIndex[0] = -1;
	maxIndex[1] = -1;
	fillFrequencyMap(numStrings, newStrings, tokenValues, frequencyMap);
	setNextToken(tokenValues, frequencyMap, maxIndex);
	int frequency = frequencyMap[maxIndex[0]][maxIndex[1]];
	for (int i = 0; i < tokenValues.size() - 1; i++) {
		free(frequencyMap[i]);
	}
	free(frequencyMap);
	return frequency;
}

BytePairTokenizer::BytePairTokenizer(int numStrings, string* strings, int maxTokens) {
	for (int i = 0; i < numStrings; i++) {
		for (int j = 0; j < strings[i].length(); j++) {
			char c = tolower(strings[i][j]);
			string token(1,c);
			if (!contains(tokenValues, token)) {
				tokenValues.emplace_back(token);
				printf("Token %d:\"%s\"\n", tokenValues.size(), token.c_str());
			}
		}
	}
	string* newStrings = (string*)malloc(numStrings * sizeof(string));
	for (int i = 0; i < numStrings; i++) {
		string convertedString = ",";
		for (int j = 0; j < strings[i].length(); j++) {
			int index = getIndex(tolower(strings[i][j]), tokenValues);
			convertedString += to_string(index) + ",";
		}
		newStrings[i] = convertedString;
	}
	int maxIndex[2];
	int frequency = calculateNextToken(numStrings, newStrings, tokenValues, maxIndex);
	while (maxIndex[0] >= 0 && maxIndex[1] >= 0 && tokenValues.size() < maxTokens) {
		tokenValues.emplace_back(tokenValues[maxIndex[0]] + tokenValues[maxIndex[1]]);
		printf("Token %d:\"%s\"  %d Occurences\n", tokenValues.size(), tokenValues[tokenValues.size() - 1].c_str(), frequency);
		replaceStrings(numStrings, newStrings, maxIndex, tokenValues.size() - 1);
		frequency = calculateNextToken(numStrings, newStrings, tokenValues, maxIndex);
	}
	printf("%d Tokens Found\n", tokenValues.size());
}

BytePairTokenizer::BytePairTokenizer(string fileName) {
	string line;
	ifstream file(fileName);
	getline(file, line);
	int num = stoi(line);
	for (int i = 0; i < num; i++) {
		printf("\r%d/%d", i, num);
		getline(file, line);
		string token(line.c_str());
		tokenValues.emplace_back(token);
	}
	printf("\r%d/%d\n", num, num);
	file.close();
}

void BytePairTokenizer::save(string fileName) {
	ofstream file(fileName.c_str());
	file << tokenValues.size() << "\n";
	for (int i = 0; i < tokenValues.size(); i++) {
		file << tokenValues[i] << "\n";
	}
	file.close();
}

float** BytePairTokenizer::tokenize(string str, int& length) {
	string convertedString = "";
	for (int i = 0; i < str.length(); i++) {
		convertedString += tolower(str[i]);
	}
	str = convertedString;
	vector<int> tokens;
	bool existingToken;
	while (str.length() > 0) {
		for (int i = tokenValues.size() - 1; i >= 0; i--) {
			string token = tokenValues[i];
			if (str.substr(0, token.length()).compare(token) == 0) {
				tokens.emplace_back(i);
				str.replace(0, token.length(), "");
				break;
			}
		}
	}
	length = tokens.size();
	float** matrix = Matrix::allocateMatrix(Matrix::ZERO_FILL, tokens.size(), tokenValues.size());
	for (int i = 0; i < tokens.size(); i++) {
		matrix[i][tokens[i]] = 1;
	}
	return matrix;
}

int* BytePairTokenizer::sparseTokenize(string str, int& length) {
	string convertedString = "";
	for (int i = 0; i < str.length(); i++) {
		convertedString += tolower(str[i]);
	}
	str = convertedString;
	vector<int> tokens;
	bool existingToken;
	while (str.length() > 0) {
		for (int i = tokenValues.size() - 1; i >= 0; i--) {
			string token = tokenValues[i];
			string substr = str.substr(0, token.length());
			if (substr.compare(token) == 0) {
				tokens.emplace_back(i);
				str.replace(0, token.length(), "");
				break;
			}
		}
	}
	length = tokens.size();
	int* sequence = new int[tokens.size()];
	for (int i = 0; i < tokens.size(); i++) {
		sequence[i] = tokens[i];
	}
	return sequence;
}

float*** BytePairTokenizer::toTokens(int numStrings, string* strings, int* numTokens) {
	float*** oneHotEmbeddings = (float***)malloc(numStrings * sizeof(float**));
	for (int i = 0; i < numStrings; i++) {
		oneHotEmbeddings[i] = tokenize(strings[i], numTokens[i]);
		printf("\r%f", 100.0 * i / numStrings);
	}
	printf("\n");
	return oneHotEmbeddings;
}

int** BytePairTokenizer::toSparseTokens(int numStrings, string* strings, int* numTokens) {
	int** sparseEmbeddings = new int* [numStrings];
	for (int i = 0; i < numStrings; i++) {
		sparseEmbeddings[i] = sparseTokenize(strings[i], numTokens[i]);
		printf("\r%f", 100.0 * i / numStrings);
	}
	printf("\n");
	return sparseEmbeddings;
}