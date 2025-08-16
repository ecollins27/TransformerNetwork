template<typename Type>
Operation* Activation::getOperation(Type& input, Type& output) {
	switch (this->activationType) {
	case ActivationType::NONE:
		return new CopyTo(input, output);
	case ActivationType::SIGMOID:
		return new SigmoidOperation(input, output);
	case ActivationType::RELU:
		return new ReluOperation(input, output);
	case ActivationType::ELU:
		return new EluOperation(alpha, input, output);
	case ActivationType::SELU:
		return new SeluOperation(input, output);
	case ActivationType::LOGLU:
		return new LogluOperation(alpha, input, output);
	case ActivationType::TANH:
		return new TanhOperation(input, output);
	case ActivationType::SWISH:
		return new SwishOperation(alpha, input, output);
	case ActivationType::SOFTMAX:
		return new SoftmaxOperation(input, output);
	default:
		throw invalid_argument("Invalid activation type");
	}
}

template<typename Type>
Operation* Activation::getDifOperation(Type& input, Type& output, Type& inputGrad, Type& outputGrad) {
	switch (this->activationType) {
	case ActivationType::NONE:
		return new CopyTo(outputGrad, inputGrad);
	case ActivationType::SIGMOID:
		return new SigmoidDifOperation(input, output, inputGrad, outputGrad);
	case ActivationType::RELU:
		return new ReluDifOperation(input, output, inputGrad, outputGrad);
	case ActivationType::ELU:
		return new EluDifOperation(alpha, input, output, inputGrad, outputGrad);
	case ActivationType::SELU:
		return new SeluDifOperation(input, output, inputGrad, outputGrad);
	case ActivationType::LOGLU:
		return new LogluDifOperation(alpha, input, output, inputGrad, outputGrad);
	case ActivationType::TANH:
		return new TanhDifOperation(input, output, inputGrad, outputGrad);
	case ActivationType::SWISH:
		return new SwishDifOperation(alpha, input, output, inputGrad, outputGrad);
	case ActivationType::SOFTMAX:
		return new SoftmaxDifOperation(input, output, inputGrad, outputGrad);
	default:
		throw invalid_argument("Invalid activation type");
	}
}