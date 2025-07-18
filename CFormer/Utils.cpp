#include "Utils.h"

int Utils::NUM_THREADS = 32;
int Utils::THREADS_PER_BLOCK = 256;
cublasHandle_t Utils::HANDLE = NULL;
bool Utils::ALLOCATE_DEVICE_MODE = false;