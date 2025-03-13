#pragma once
#include "Input2D.h"
#include "Loss1D.h"
#include <climits>
#include <thread>
#include <functional>
#include "Dataset.h"
#include "LinformerAttention.h"

class Model2DTo1D {

public:

	static int NUM_CORES;

	Input2D* inputLayer;
	Layer* tempLayer;
	Layer1D* outputLayer;
	int t;

	const int MAX_NUM_TOKENS = 200;

	Model2DTo1D(int inputSize);

	void addLayer(Layer* layer);

	template<class T>
	T* getLayer(int index) {
		Layer* layer = inputLayer;
		for (int i = 0; i < index; i++) {
			layer = layer->nextLayer;
		}
		return (T*)layer;
	}

	int getNumParameters();
	void addTransformer(int numHeads, int keySize, int valueSize);
	void addLinformer(int numHeads, int keySize, int valueSize, int projSize);

	void fit(Loss1D* lossFunction, Dataset* data, int numMetrics, Loss1D** metrics, TrainingParams* params);
	void test(Loss1D* lossFunction, Dataset* data, int numMetrics, Loss1D** metrics);
	void save(string fileName);

private:
	void applyGradients(float learningRate);
	void updateAverages(Loss1D* lossFunction, float** y, float* averages, int numMetrics, Loss1D** metrics);
	void predict(void* input, bool sparse, int thread);
	void evaluateValidation(Loss1D* lossFunction, Dataset* valData, int batchSize, int numMetrics, Loss1D** metrics);
	void forwardPropagate(void* input, bool sparse, int thread);
	void backPropagate(Loss1D* lossFunction, int thread);

	Dataset* partitionData(Dataset* data);
};

