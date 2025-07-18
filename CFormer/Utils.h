#pragma once
#include <cublas_v2.h>
#include <iostream>

using namespace std;

class Utils {

public:
	static int NUM_THREADS;
	static int THREADS_PER_BLOCK;
	static cublasHandle_t HANDLE;
	static bool ALLOCATE_DEVICE_MODE;

};

