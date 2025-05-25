echo "== Copying Matrix2.cpp to Matrix2.cu =="
cp Matrix2.cpp Matrix2.cu

echo "== Compiling CUDA source =="
nvcc -std=c++20 -c Matrix2.cu -o Matrix2.o

echo "== Compiling other C++ files =="
CPP_SOURCES=$(ls *.cpp | grep -v Matrix2.cpp)
g++ -w -std=c++2b -O2 -fexceptions -Wall -Wextra -Wno-unused-parameter -DNDEBUG -D_CONSOLE -D_UNICODE -DUNICODE -fno-strict-aliasing -Wno-sign-compare -ffp-contract=off -fPIC -g -c $CPP_SOURCES

echo "== Linking all object files =="
nvcc *.o -o main

echo "== Running program =="
./main