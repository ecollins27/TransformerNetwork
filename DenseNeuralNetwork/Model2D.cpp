#include "Model2D.h"
#include "ResidualSave2D.h"
#include "ResidualAdd2D.h"
#include "LayerNormalization2D.h"
#include "Dense2D.h"
#include "MultiHeadAttention.h"

int Model2D::NUM_CORES = 1;

Model2D::Model2D(int inputSize) {
	inputLayer = new Input2D(inputSize);
	outputLayer = inputLayer;
	t = 0;
	output1D = false;
}

void Model2D::addLayer(Layer* layer) {
	layer->setPrevLayer(outputLayer);
	outputLayer->setNextLayer(layer);
	outputLayer = layer;
	output1D = Layer::instanceOf<Layer1D>(outputLayer);
}

void Model2D::applyGradients(float learningRate) {
	t++;
	inputLayer->applyGradients(learningRate, t);
}

int Model2D::getNumParameters() {
	return inputLayer->getNumParameters();
}

void Model2D::predict(float** input, int thread) {
	inputLayer->setInput(thread, input);
	inputLayer->predict(thread);
}

void Model2D::forwardPropagate(float** input, int thread) {
	inputLayer->setInput(thread, input);
	inputLayer->forwardPropagate(thread);
}

void Model2D::backPropagate(Loss* lossFunction, float** yTrue, int thread) {
	if (thread == 0) {
		lossFunction->differentiate(outputLayer, yTrue, thread, output1D);
	}
	outputLayer->backPropagate(thread);
}

void Model2D::addTransformerBlock(int numHeads, int keySize, int valueSize) {
	int size = outputLayer->size;
	ResidualSave2D* rs1 = { new ResidualSave2D() };
	this->addLayer(rs1);
	this->addLayer({ new MultiHeadAttention(numHeads, keySize, valueSize) });
	this->addLayer({ new ResidualAdd2D(rs1) });
	this->addLayer({ new LayerNormalization2D() });
	ResidualSave2D* rs2 = { new ResidualSave2D() };
	this->addLayer(rs2);
	this->addLayer({ new Dense2D(Activation::SWISH, size) });
	this->addLayer({ new Dense2D(Activation::SWISH, size) });
	this->addLayer({ new ResidualAdd2D(rs2) });
	this->addLayer({ new LayerNormalization2D() });
}

void Model2D::fitPoint(Loss* lossFunction, float** X, float** y, int thread) {
	forwardPropagate(X, thread);
	backPropagate(lossFunction, y, thread);
}

void Model2D::threadFit(Model2D* model, Loss* lossFunction, int trainingNum, int* numTokens, float*** X, float*** y, float* losses, int thread, int* batchProgress, int numMetrics, Loss** metrics, int batchSize, float learningRate, int epoch, int numEpochs) {
	for (int i = thread; i < trainingNum; i += batchSize) {
		if (thread == 0) {
			printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, numEpochs, i, trainingNum, losses[numMetrics] / i);
			for (int j = 0; j < numMetrics; j++) {
				printf("%s:%f  ", metrics[j]->toString().c_str(), losses[j] / i);
			}
			model->inputLayer->setNumTokens(&numTokens[i]);
			*batchProgress = 0;
		}
		else {
			while (*batchProgress != 0){
				printf("");
			}
		}
		if (model->output1D) {
			model->fitPoint(lossFunction, X[i], &y[0][i - thread], thread);
		} else {
			model->fitPoint(lossFunction, X[i], y[i], thread);
			for (int j = 0; j < numMetrics; j++) {
				losses[j] += metrics[j]->loss(model->outputLayer, y[i], thread, model->output1D);
			}
			losses[numMetrics] += lossFunction->loss(model->outputLayer, y[i], thread, model->output1D);
		}
		(*batchProgress)++;
		if (thread == 0) {
			while (*batchProgress < batchSize){}
			if (model->output1D) {
				for (int j = 0; j < numMetrics; j++) {
					losses[j] += metrics[j]->loss(model->outputLayer, &y[0][i], 0, model->output1D);
				}
				losses[numMetrics] += lossFunction->loss(model->outputLayer, &y[0][i], 0, model->output1D);
			}
			model->applyGradients(learningRate);
		}
	}
	if (thread == 0) {
		*batchProgress = -1;
	}
	printf("\nEnding Thread %d", thread);
}

void Model2D::oneThreadFit(Loss* lossFunction, int numData, int* numTokens, float*** X, float*** y, int numMetrics, Loss** metrics, TrainingParams* params, string filename) {
	int maxTokenSize = 0;
	float minLoss = INT_MAX;
	for (int i = 0; i < numData; i++) {
		if (numTokens[i] > maxTokenSize) {
			maxTokenSize = numTokens[i];
		}
	}
	printf("MaxTokenSize:%d\n", maxTokenSize);
	inputLayer->setMaxNumTokens(maxTokenSize);
	float valSplit = params->get<float>(TrainingParams::VAL_SPLIT);
	int batchSize = params->get<int>(TrainingParams::BATCH_SIZE);
	int numEpochs = params->get<int>(TrainingParams::NUM_EPOCHS);
	float learningRate = params->get<float>(TrainingParams::LEARNING_RATE);
	inputLayer->setOptimizer(params->get<Optimizer*>(TrainingParams::OPTIMIZER));
	float* averages = NULL;
	averages = new float[numMetrics + 1];
	int trainingNum = (int)(numData * (1 - valSplit));
	trainingNum -= trainingNum % batchSize;
	inputLayer->setBatchSize(batchSize);
	for (int epoch = 0; epoch < numEpochs; epoch++) {
		shuffle(numData, numTokens, X, y);
		for (int i = 0; i < numMetrics + 1; i++) {
			averages[i] = 0;
		}
		for (int i = 0; i < trainingNum; i += batchSize) {
			printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, numEpochs, i, trainingNum, averages[numMetrics] / i);
			for (int j = 0; j < numMetrics; j++) {
				printf("%s:%f  ", metrics[j]->toString().c_str(), averages[j] / i);
			}
			inputLayer->setNumTokens(&numTokens[i]);
			for (int k = 0; k < batchSize; k++) {
				forwardPropagate(X[i + k], k);
			}
			for (int k = 0; k < batchSize; k++) {
				if (output1D){
					backPropagate(lossFunction, &y[0][i], k);
				}
				else {
					backPropagate(lossFunction, y[i + k], k);
					for (int j = 0; j < numMetrics; j++) {
						averages[j] += metrics[j]->loss(outputLayer, y[i + k], k, output1D);
					}
					averages[numMetrics] += lossFunction->loss(outputLayer, y[i + k], k, output1D);
				}
			}
			if (output1D) {
				for (int j = 0; j < numMetrics; j++) {
					averages[j] += metrics[j]->loss(outputLayer, &y[0][i], 0, output1D);
				}
				averages[numMetrics] += lossFunction->loss(outputLayer, &y[0][i], 0, output1D);
			}
			applyGradients(learningRate);
		}
		int valNum = (int)(numData * valSplit);
		if (valNum > 0) {
			for (int i = 0; i < numMetrics + 1; i++) {
				averages[i] = 0;
			}
			int i = trainingNum;
			for (i = trainingNum; i < numData - numData % batchSize; i += batchSize) {
				inputLayer->setNumTokens(&numTokens[i]);
				for (int k = 0; k < batchSize; k++) {
					predict(X[i + k], k);
					if (!output1D) {
						for (int j = 0; j < numMetrics; j++) {
							averages[j] += metrics[j]->loss(outputLayer, y[i + k], k, output1D);
						}
						averages[numMetrics] += lossFunction->loss(outputLayer, y[i + k], k, output1D);
					}
				}
				if (output1D) {
					for (int j = 0; j < numMetrics; j++) {
						averages[j] += metrics[j]->loss(outputLayer, &y[0][i], 0, output1D);
					}
					averages[numMetrics] += lossFunction->loss(outputLayer, &y[0][i], 0, output1D);
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

void Model2D::fit(Loss* lossFunction, int numData, int* numTokens, float*** X, float*** y, int numMetrics, Loss** metrics, TrainingParams* params, string filename) {
	int maxTokenSize = 0;
	float minLoss = INT_MAX;
	for (int i = 0; i < numData; i++) {
		if (numTokens[i] > maxTokenSize) {
			maxTokenSize = numTokens[i];
		}
	}
	printf("MaxTokenSize:%d\n", maxTokenSize);
	inputLayer->setMaxNumTokens(maxTokenSize);
	float valSplit = params->get<float>(TrainingParams::VAL_SPLIT);
	int batchSize = params->get<int>(TrainingParams::BATCH_SIZE);
	int numEpochs = params->get<int>(TrainingParams::NUM_EPOCHS);
	float learningRate = params->get<float>(TrainingParams::LEARNING_RATE);
	inputLayer->setOptimizer(params->get<Optimizer*>(TrainingParams::OPTIMIZER));
	float* averages = NULL;
	averages = new float[numMetrics + 1];
	int trainingNum = (int)(numData * (1 - valSplit));
	trainingNum -= trainingNum % batchSize;
	inputLayer->setBatchSize(batchSize);
	int* batchProgress = new int(0);
	for (int epoch = 0; epoch < numEpochs; epoch++) {
		*batchProgress = 1;
		thread* threads = new thread[batchSize];
		int i = 0;
		for (i = 0; i < batchSize; i++) {
			threads[i] = thread(&Model2D::threadFit, this, lossFunction, trainingNum, numTokens, X, y, averages, i, batchProgress, numMetrics, metrics, batchSize, learningRate, epoch, numEpochs);
		}
		while (*batchProgress != -1){
			printf("");
		}
	}
}

void Model2D::test(Loss* lossFunction, int numData, int* numTokens, float*** X, float*** y, int numMetrics, Loss** metrics) {
	int maxTokenSize = 0;
	for (int i = 0; i < numData; i++) {
		if (numTokens[i] > maxTokenSize) {
			maxTokenSize = numTokens[i];
		}
	}
	printf("MaxTokenSize:%d\n", maxTokenSize);
	inputLayer->setMaxNumTokens(maxTokenSize);
	inputLayer->setBatchSize(1);
	float* averages = new float[numMetrics + 1];
	for (int i = 0; i < numData; i++) {
		inputLayer->setNumTokens(&numTokens[i]);
		predict(X[i], 0);
		averages[numMetrics] += lossFunction->loss(outputLayer, output1D ? &y[0][i] : y[i], 0, output1D);
		for (int j = 0; j < numMetrics; j++) {
			averages[j] += metrics[j]->loss(outputLayer, output1D ? &y[0][i] : y[i], 0, output1D);
		}
		printf("\rTest %d/%d  NumTokens:%d  TestLoss:%f  ", i + 1, numData, numTokens[i], averages[numMetrics] / (i + 1));
		for (int j = 0; j < numMetrics; j++) {
			printf("Test%s:%f  ", metrics[j]->toString().c_str(), averages[j] / (i + 1));
		}
	}
	printf("\n");
}

void Model2D::shuffle(int numData, int* numTokens, float*** X, float*** y) {
	for (int i = 0; i < numData; i++) {
		int index = (int)(numData * ((float)rand() / (RAND_MAX + 1)));
		swap(X[i], X[index]);
		if (output1D) {
			swap(y[0][i], y[0][index]);
		}
		else {
			swap(y[i], y[index]);
		}
		swap(numTokens[i], numTokens[index]);
	}
}

void Model2D::save(string filename) {
	ofstream file(filename.c_str());
	file << "TransformerModel\n";
	inputLayer->save(file);
	file.close();
}