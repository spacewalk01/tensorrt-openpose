#include <cuda_runtime.h>

int main() {
	int* a;
	cudaMalloc(&a, 100);

	cudaFree(a);

	return 0;
}