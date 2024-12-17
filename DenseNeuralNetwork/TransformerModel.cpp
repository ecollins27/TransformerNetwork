#include "TransformerModel.h"

TransformerModel::TransformerModel(int inputSize) {
	inputLayer = { new InputLayer(inputSize) };
	outputLayer = inputLayer;
	t = 0;
}

void TransformerModel::predict(float** input) {
	inputLayer->setInput(input);
	inputLayer->predict();
}

void TransformerModel::forwardPropagate(float** input) {
	inputLayer->setInput(input);
	inputLayer->forwardPropagate();
}

void TransformerModel::backPropagate(Loss* lossFunction, float* yTrue) {
	lossFunction->differentiate(outputLayer, &yTrue);
	outputLayer->backPropagate();
}

void TransformerModel::addTransformerBlock(int numHeads, int keySize, int valueSize) {
	int size = outputLayer->size;
	ResidualSave* rs1 = { new ResidualSave() };
	this->addLayer(rs1);
	this->addLayer({ new MultiHeadAttentionLayer(numHeads, keySize, valueSize) });
	this->addLayer({ new ResidualAdd(rs1) });
	this->addLayer({ new LayerNormalization() });
	ResidualSave* rs2 = { new ResidualSave() };
	this->addLayer(rs2);
	this->addLayer({ new DenseLayer(Activation::SWISH, size) });
	this->addLayer({ new DenseLayer(Activation::SWISH, size) });
	this->addLayer({ new ResidualAdd(rs2) });
	this->addLayer({ new LayerNormalization() });
}

void TransformerModel::fit(Loss* lossFunction, int numTokens, float** X, float* y, float* losses, int numMetrics, Loss** metrics, TrainingParams* params) {
	inputLayer->setBatchSize(numTokens);
	forwardPropagate(X);
	backPropagate(lossFunction, y);
	for (int i = 0; i < numMetrics; i++) {
		losses[i] += metrics[i]->loss(outputLayer, &y);
	}
	losses[numMetrics] += lossFunction->loss(outputLayer, &y);
}

void TransformerModel::fit(Loss* lossFunction, int numData, int* numTokens, float*** X, float** y, int numMetrics, Loss** metrics, TrainingParams* params, string filename) {
	int maxTokenSize = 0;
	float minLoss = INT_MAX;
	for (int i = 0; i < numData; i++) {
		if (numTokens[i] > maxTokenSize) {
			maxTokenSize = numTokens[i];
		}
	}
	printf("MaxTokenSize:%d\n", maxTokenSize);
	inputLayer->setMaxBatchSize(maxTokenSize);
	float valSplit = params->get<float>(TrainingParams::VAL_SPLIT);
	int batchSize = 1;
	int numEpochs = params->get<int>(TrainingParams::NUM_EPOCHS);
	float learningRate = params->get<float>(TrainingParams::LEARNING_RATE);
	inputLayer->setOptimizer(params->get<Optimizer*>(TrainingParams::OPTIMIZER));
	float* averages = NULL;
	averages = new float[numMetrics + 1];
	int trainingNum = (int)(numData * (1 - valSplit));
	for (int epoch = 0; epoch < numEpochs; epoch++) {
		shuffle(numData, numTokens, X, y);
		for (int i = 0; i < numMetrics + 1; i++) {
			averages[i] = 0;
		}
		for (int i = 0; i < trainingNum; i++) {
			fit(lossFunction, numTokens[i], X[i], y[i], averages, numMetrics, metrics, params);
			applyGradients(learningRate);
			printf("\rEpoch %d/%d  %d/%d  NumTokens:%d  Loss:%f  ", epoch + 1, numEpochs, i + 1, trainingNum, numTokens[i], averages[numMetrics] / (i + 1));
			for (int j = 0; j < numMetrics; j++) {
				printf("%s:%f  ", metrics[j]->toString().c_str(), averages[j] / (i + 1));
			}
		}
		int valNum = (int)(numData * valSplit);
		if (valNum > 0) {
			for (int i = 0; i < numMetrics + 1; i++) {
				averages[i] = 0;
			}
			int i = trainingNum;
			for (i = trainingNum; i < numData; i ++) {
				inputLayer->setBatchSize(numTokens[i]);
				predict(X[i]);
				averages[numMetrics] += lossFunction->loss(outputLayer, &y[i]);
				for (int j = 0; j < numMetrics; j++) {
					averages[j] += metrics[j]->loss(outputLayer, &y[i]);
				}
			}
			float valLoss = averages[numMetrics] / (numData - trainingNum);
			printf("ValLoss:%f ", valLoss);
			for (i = 0; i < numMetrics; i++) {
				printf("Val%s:%f  ", metrics[i]->toString().c_str(), averages[i] / (numData - trainingNum));
			}
			if (valLoss < minLoss) {
				minLoss = valLoss;
				printf("\nSaving Model . . .");
				this->save(filename);
			}
		}
		printf("\n");
	}
}

void TransformerModel::test(Loss* lossFunction, int numData, int* numTokens, float*** X, float** y, int numMetrics, Loss** metrics) {
	int maxTokenSize = 0;
	for (int i = 0; i < numData; i++) {
		if (numTokens[i] > maxTokenSize) {
			maxTokenSize = numTokens[i];
		}
	}
	printf("MaxTokenSize:%d\n", maxTokenSize);
	inputLayer->setMaxBatchSize(maxTokenSize);
	float* averages = new float[numMetrics + 1];
	for (int i = 0; i < numData; i++) {
		inputLayer->setBatchSize(numTokens[i]);
		predict(X[i]);
		averages[numMetrics] += lossFunction->loss(outputLayer, &y[i]);
		for (int j = 0; j < numMetrics; j++) {
			averages[j] += metrics[j]->loss(outputLayer, &y[i]);
		}
		printf("\rTest %d/%d  NumTokens:%d  TestLoss:%f  ", i + 1, numData, numTokens[i], averages[numMetrics] / (i + 1));
		for (int j = 0; j < numMetrics; j++) {
			printf("Test%s:%f  ", metrics[j]->toString().c_str(), averages[j] / (i + 1));
		}
	}
	printf("\n");
}

void TransformerModel::shuffle(int numData, int* numTokens, float*** X, float** y) {
	for (int i = 0; i < numData; i++) {
		int index = (int)(numData * ((float)rand() / (RAND_MAX + 1)));
		swap(X[i], X[index]);
		swap(y[i], y[index]);
		swap(numTokens[i], numTokens[index]);
	}
}

void TransformerModel::save(string filename) {
	ofstream file(filename.c_str());
	file << "TransformerModel\n";
	inputLayer->save(file);
	file.close();
}