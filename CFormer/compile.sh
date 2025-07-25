echo "== Cleaning up C++ files =="
dos2unix -q *.h
dos2unix -q *.cpp

echo "== Copying CUDA Source to cu files =="
CU_SOURCES=($(grep -l 'include "MatrixKernel.h"' *.cpp))
CU_SOURCES+=("Matrix2.cpp")
CU_SOURCES+=("MatrixBatch.cpp")
CU_SOURCES+=("PropagationQueue.cpp")
CU_SOURCES+=("MatrixOperations.cpp")
CU_SOURCES+=("MatrixBatchOperations.cpp")
echo "  CU_SOURCES: ${CU_SOURCES[*]}"
for file in "${CU_SOURCES[@]}";
do cp "$file" "${file%.cpp}.cu"
done

echo "== Compiling CUDA source =="
nvcc -w -std=c++20 -lcublas -c *.cu

echo "== Compiling other C++ files =="
ALL_CPP=(*.cpp)
CPP_SOURCES=()
for file in "${ALL_CPP[@]}";
do if [[ ! "${CU_SOURCES[@]}" =~ "$file" ]];
then CPP_SOURCES+=("$file")
fi
done
g++ -w -std=c++2b -O2 -fexceptions -Wall -Wextra -Wno-unused-parameter -DNDEBUG -D_CONSOLE -D_UNICODE -DUNICODE -fno-strict-aliasing -Wno-sign-compare -ffp-contract=off -fPIC -g -c ${CPP_SOURCES[*]}

echo "== Linking all object files =="
nvcc *.o -lcublas -o main

echo "== Deleting extra files =="
rm *.o
rm *.cu
rm d2*

echo "== Running program =="
gdb ./main
