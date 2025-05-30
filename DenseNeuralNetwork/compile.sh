echo "== Cleaning up C++ files =="
dos2unix -q *.cpp

echo "== Copying CUDA Source to cu files =="
cp Matrix2.cpp Matrix2.cu
cp MatrixBatch.cpp MatrixBatch.cu

echo "== Compiling CUDA source =="
nvcc -std=c++20 -c *.cu

echo "== Compiling other C++ files =="
CPP_SOURCES=$(ls *.cpp | grep -v Matrix2.cpp)
g++ -w -std=c++2b -O2 -fexceptions -Wall -Wextra -Wno-unused-parameter -DNDEBUG -D_CONSOLE -D_UNICODE -DUNICODE -fno-strict-aliasing -Wno-sign-compare -ffp-contract=off -fPIC -g -c $CPP_SOURCES

echo "== Linking all object files =="
nvcc *.o -o main

echo "== Running program =="
./main