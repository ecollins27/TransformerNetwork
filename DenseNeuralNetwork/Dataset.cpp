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
	for (int i = 0; i < numData; i++) {
		int index = (int)(numData * ((float)rand() / (RAND_MAX + 1)));
		swap(X[i], X[index]);
		swap(y[i], y[index]);
		swap(numTokens[i], numTokens[index]);
	}
}