#include "Dataset.h"

Dataset* Dataset::getMiniBatch(int index, int batchSize) {
	return new Dataset(batchSize, &numTokens[index], &X[index], &y[index], sparseX);
}

int Dataset::getMaxNumTokens() {
	if (numTokens == NULL) {
		return 1;
	}
	int maxNumTokens = 0;
	for (int i = 0; i < numData; i++) {
		if (numTokens[i] > maxNumTokens) {
			maxNumTokens = numTokens[i];
		}
	}
	return maxNumTokens;
}

void Dataset::shuffle() {
	uniform_real_distribution<float> distribution = uniform_real_distribution<float>(0, numData);
	default_random_engine generator;
	for (int i = 0; i < numData; i++) {
		int index = (int)distribution(generator);
		swap(X[i], X[index]);
		swap(y[i], y[index]);
		swap(numTokens[i], numTokens[index]);
	}
}