#include "BytePairTokenizer.h"

bool contains(vector<string> strings, string str) {
	for (string s : strings) {
		if (s.compare(str) == 0) {
			//printf("%s %s\n", s.c_str(), str.c_str());
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

void calculateNextToken(int numStrings, string* newStrings, vector<string> tokenValues, int maxIndex[2]) {
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
	for (int i = 0; i < tokenValues.size() - 1; i++) {
		free(frequencyMap[i]);
	}
	free(frequencyMap);
}

BytePairTokenizer::BytePairTokenizer(int numStrings, string* strings) {
	for (int i = 0; i < numStrings; i++) {
		for (int j = 0; j < strings[i].length(); j++) {
			char c = tolower(strings[i][j]);
			string token(1,c);
			if (!contains(tokenValues, token)) {
				tokenValues.emplace_back(token);
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
	calculateNextToken(numStrings, newStrings, tokenValues, maxIndex);
	while (maxIndex[0] >= 0 && maxIndex[1] >= 0 && tokenValues.size() < 1000) {
		tokenValues.emplace_back(tokenValues[maxIndex[0]] + tokenValues[maxIndex[1]]);
		replaceStrings(numStrings, newStrings, maxIndex, tokenValues.size() - 1);
		printf("\"%s\" %d\n", tokenValues[tokenValues.size() - 1], tokenValues.size() - 1);
		calculateNextToken(numStrings, newStrings, tokenValues, maxIndex);
	}
}

BytePairTokenizer::BytePairTokenizer(string fileName) {
	string line;
	ifstream file(fileName);
	getline(file, line);
	int num = stoi(line);
	for (int i = 0; i < num; i++) {
		getline(file, line);
		string token(line.c_str());
		tokenValues.emplace_back(token);
	}
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

float** BytePairTokenizer::tokenize(string str) {
	string convertedString = "";
	for (int i = 0; i < str.length(); i++) {
		convertedString += tolower(str[i]);
	}
	str = convertedString;
	vector<int> tokens;
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
	for (int i = 0; i < tokens.size(); i++) {
		printf("%d,", tokens[i]);
	}
	printf("\n");
	for (int i = 0; i < tokens.size(); i++) {
		printf("%s ", tokenValues[tokens[i]].c_str());
	}
	printf("\n");
	float** matrix = Matrix::allocateMatrix(Matrix::ZERO_FILL, tokens.size(), tokenValues.size());
	for (int i = 0; i < tokens.size(); i++) {
		matrix[i][tokens[i]] = 1;
	}
	return matrix;
}