echo "== Cleaning up C++ files =="
dos2unix -q *.h
dos2unix -q *.cpp

echo "== Copying CUDA Source to cu files =="
cp Matrix2.cpp Matrix2.cu
cp MatrixBatch.cpp MatrixBatch.cu
cp main.cpp main.cu

echo "== Compiling CUDA source =="
nvcc -w -std=c++20 -lcublas -c *.cu

echo "== Compiling other C++ files =="
CPP_SOURCES=$(ls *.cpp | grep -Ev "(Matrix2|MatrixBatch|main).cpp")
g++ -w -std=c++2b -O2 -fexceptions -Wall -Wextra -Wno-unused-parameter -DNDEBUG -D_CONSOLE -D_UNICODE -DUNICODE -fno-strict-aliasing -Wno-sign-compare -ffp-contract=off -fPIC -g -c $CPP_SOURCES

echo "== Linking all object files =="
nvcc *.o -lcublas -o main

echo "== Deleting extra files =="
rm *.o
rm *.cu
rm d2*

echo "== Running program =="
./main
