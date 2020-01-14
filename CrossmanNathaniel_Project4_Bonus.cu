
/**
*@copyright     All rights are reserved, this code/project is not Open Source or Free
*@author        Nathaniel Crossman (U00828694)
*@email		 crossman.4@wright.edu
*
*@Professor     Meilin Liu
*@Course_Number CS 4370/6370-90
*@Date			 12 5, 2019
*
Project Name:  CrossmanNathaniel_Project_Bonus WORKS
•	Task 1 - 
	o	Basic CUDA Program using global memory
•	Task 2 – 
	o	CUDA program that takes advantage of shared memory
*/
// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string.h>

//CUDA runtime
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
//#include <helper_functions.h>
#include <cuda_runtime_api.h>
//#include <helper_cuda.h>

//For right now we will set Block size to a fixed value.
#define BLOCK_SIZE 256
#define HISTOGRAM_SIZE 256

/*
getAnyErrors
This F() is a better way of showing if any errors happen.
I removed all inline error checking.
*/
#define getAnyErrors(msg) \
    do { \
        cudaError_t myErrorList = cudaGetLastError(); \
        if (myErrorList != cudaSuccess) { \
			printf(" Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(myErrorList),__FILE__, __LINE__ );\
        } \
    } while (0)


/*
Work Inefficient Histogram kernel
Reference:
Our Textbook: CUDA by Example

Develop a CUDA program with GPU threads collectively performing the histogram calculation. 
Use an atomic instruction to enforce one thread at a time accessing to individual locations in the global histogram array.
*/
__global__ void histogram_Atomic_kernel(int *buffer, unsigned int *histo, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (index < n) {
		atomicAdd(&(histo[buffer[index]]), 1);
		
		/*
		The call atomicAdd( addr, y ); generates an atomic sequence of operations
that read the value at address addr, adds y to that value, and stores the
result back to the memory address addr. The hardware guarantees us that no
other thread can read or write the value at address addr while we perform these
operations, thus ensuring predictable results.
		*/
		
		/*
		In our example, the address in
question is the location of the histogram bin that corresponds to the current byte.
If the current byte is buffer[i], just like we saw in the CPU version, the corresponding
histogram bin is histo[buffer[i]]. The atomic operation needs the
address of this bin, so the first argument is therefore &(histo[buffer[i]]).
Since we simply want to increment the value in that bin by one, the second argument
is 1.
		*/
		
/*Essentially, when thousands
of threads are trying to access a handful of memory locations, a great deal of
contention for our 256 histogram bins can occur. To ensure atomicity of the increment
operations, the hardware needs to serialize operations to the same memory
location. This can result in a long queue of pending operations, and any performance
gain we might have had will vanish*/
		index += stride;
	}
}

/*
Work Efficient Parallel Reduction Kernel with Dynamic Parallelism 
Reference:
Our Textbook: CUDA by Example
*/
__global__ void histogram_kernel(int *buffer, unsigned int *histo, int n) {
	__shared__ unsigned int tempHistogram[HISTOGRAM_SIZE];
	tempHistogram[threadIdx.x] = 0;
	__syncthreads();

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	while (index < n) {
		atomicAdd(&tempHistogram[buffer[index]], 1);
		index += offset;
	}
	__syncthreads();
	atomicAdd(&(histo[threadIdx.x]), tempHistogram[threadIdx.x]);
}



//prototypes
void cls();
int debugOn();
int menuShow();
void getSizeOfN(int &size);
void mainSwitch(int option);
void prefixSumMain(int choice);
void printf_matrix(unsigned int *A, int size);
void setDataValues(int* arrayValue, int size);
void verify(int resultOfHost, int resultofGpu);
void printRandoms(int* arrayValue, int size, int lower, int upper);
void HistogramCPU_R(unsigned int * histo, int* buffer, int sizeOfN, double &cpu_time);


int main(){	
	while (true) {
		mainSwitch(menuShow());
		printf("\n");
	}
	return 0;
}

int menuShow() {
	int hold;
	do {
		printf("1. Basic CUDA Program using global memory - Task 1\n");
		printf("2. CUDA program that takes advantage of shared memory - Task 2\n");
		printf("3. Quit\n");
		printf("---------------------------------------\n");
		printf("Enter Choice: ");
		scanf("%d", &hold);

		if (hold < 1 || hold > 3) {
			cls();
		}
	} while (hold < 1 || hold > 3);
	return hold;
}

void cls() {
	for (int i = 0; i < 30; ++i)
			printf("\n");
	system("@cls||clear");
}

void mainSwitch(int option) {
	switch (option) {
	case 1:
		prefixSumMain(option);
		break;
	case 2:
		prefixSumMain(option);
		break;
	case 3:
		exit(0);
		break;
	}
}

void getSizeOfN(int &size) {
	printf("Please specify the size of N (N = #elements) \n");
	printf("For example, you could enter 131072 and the size would be (131072 )\n");
	printf("OR, you could enter 1048576 ‬ and the size would be 1048576\n");
	printf("Enter Size:");
	scanf("%d", &size);
	cls();
}


void printf_matrix(unsigned int *A, int size) {
	int i;
	for (i = 0; i < size; ++i) {
		printf("%d \t", A[i]);
	}
	printf("\n");
}

void verify(int resultOfHost, int resultofGpu) {
	if (resultOfHost == resultofGpu)
		printf("The Test Passed\n");
	else
		printf("The Test failed\n");
}

int debugOn() {
	int hold;
	do {
		printf("\nRun in debug mode?\n");
		printf("Debug mode prints out alot of helpful info,\nbut it can takes a long time with big matrixes\n");
		printf("Enter 1 for Yes and 0 for No:");
		scanf("%d", &hold);
		if (hold < 0 || hold > 1) {
			cls();
		}
	} while (hold < 0 || hold > 1); 
	cls();
	return hold;
}

void setDataValues(int* arrayValue, int size)
{
	int init = 1325;
	int i = 0;
	for (; i < size; ++i) {
		init = 3125 * init % 65537;
		arrayValue[i] = init % 256;
	}
	
}
//for testing only. need to see why all are 512 
void printRandoms(int* arrayValue, int size, int lower, int upper)
{
	int i;
	for (i = 0; i < size; ++i) {
		arrayValue[i] = (rand() % (upper - lower + 1)) + lower;
		
	}
}

void HistogramCPU_R(unsigned int * histo, int* buffer, int sizeOfN, double &cpu_time) {
	printf("CPU doing work....\n");
	clock_t startTime, endTime;
	startTime = clock();
	int i;
	for (i = 0; i < sizeOfN; i++) {
		histo[buffer[i]]++;
	}
	endTime = clock();
	cpu_time = ((double)(endTime - startTime))* 1000.0 / CLOCKS_PER_SEC;
}

//Check answer
void verify(unsigned int *x, unsigned int *y, int n) {
	for (int i = 0; i< n; i++) {
		if (x[i] != y[i]) {
			printf("TEST FAILED\n");
			return;
		}
	}
	printf("TEST PASSED \n");
}


void prefixSumMain(int choice) {
	int sizeOfN		= 0;
	float secTotal	= 0.0f;
	double cpu_time = 0.0f;
	float temp_time = 0;
	int blocksize	= BLOCK_SIZE;

	unsigned int *h_finalAnswerFromCPU	; //This hold final histo
	unsigned int *h_finalAnswerFromGPU	; //This hold final histo
	int *h_array_data_input_CPU	; ////This hold input data


	int *d_array_data_input;
	unsigned int *d_array_data_output;

	//int booleanValue = debugOn();
	getSizeOfN(sizeOfN);

	size_t dsize = sizeOfN * sizeof(int);
	size_t dsize_histo = HISTOGRAM_SIZE * sizeof(unsigned int);

	h_finalAnswerFromCPU	= (unsigned int*)malloc(dsize_histo);
	h_finalAnswerFromGPU	= (unsigned int*)malloc(dsize_histo);
	h_array_data_input_CPU	= (int*)malloc(dsize);

	memset(h_array_data_input_CPU, 0, dsize);
	memset(h_finalAnswerFromCPU, 0, dsize_histo);
	memset(h_finalAnswerFromGPU, 0, dsize_histo);
	
	printf("ElementSize: %d \nSize of Thread block: %d", sizeOfN, BLOCK_SIZE);
	printf("\n\n");


	if (h_array_data_input_CPU == NULL ||h_finalAnswerFromCPU == NULL || h_finalAnswerFromGPU == NULL) {
		printf("Failed to allocate host matrix C!\n");
	}

	//Set Histogram
	setDataValues(h_array_data_input_CPU, sizeOfN);
	//printRandoms(h_array_data_input_CPU, sizeOfN, 0, 255);
	
	
	//Allocating memory 
	cudaMalloc((void **)(&d_array_data_input), dsize);
	getAnyErrors("Allocating memory for d_array_data_input ");
	//Copy Value to GPU 
	cudaMemcpy(d_array_data_input, h_array_data_input_CPU, dsize, cudaMemcpyHostToDevice);
	getAnyErrors("Copying array data from host to deviced ");

	//Allocating memory 
	cudaMalloc((void **)(&d_array_data_output), dsize_histo);
	getAnyErrors("Allocating memory for d_array_data_output ");
	//Setting Value to GPU 
	cudaMemset(d_array_data_output, 0, dsize_histo);
	getAnyErrors("Set all item in histo to 0");


	printf("CPU is do working....\n");
	printf("***********************************************************************\n");
	clock_t startTime, endTime;
	startTime = clock();

	HistogramCPU_R(h_finalAnswerFromCPU, h_array_data_input_CPU, sizeOfN, cpu_time);

	endTime = clock();
	cpu_time = ((double)(endTime - startTime))* 1000.0 / CLOCKS_PER_SEC;

	//int myDynamicGrid = (sizeOfN - 1) / (BLOCK_SIZE * 2) + 1;
	//int myDynamicGrid = sizeOfN / (BLOCK_SIZE);
	int myDynamicGrid = blocksize * 2;
	//Crucial kernel configurations
	dim3 dimBlock(blocksize);
	dim3 dimGrid(myDynamicGrid);

	printf("\nGPU is working \n");
	printf("***********************************************************************\n");
	printf("Kernel Starts\n");
	printf("-------------------------------------------------------------------\n");
	printf("Kernel Parameter\n");
	printf("dimBlock :%d\n", blocksize);
	printf("dimGrid :%d\n", myDynamicGrid);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	if (choice == 1) {
		printf("\nRunning histogram_Atomic_kernel - Task 1 \n");
		histogram_Atomic_kernel << < dimGrid, dimBlock >> > (d_array_data_input, d_array_data_output, sizeOfN);
	}
	if(choice == 2){
		printf("\nRunning histogram_kernel - Task 2 \n");
		histogram_kernel << < dimGrid, dimBlock >> >(d_array_data_input, d_array_data_output, sizeOfN);
	}
	getAnyErrors("First Call");
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&temp_time, start, stop);
	secTotal = secTotal + temp_time;
	cudaEventDestroy(start);
	cudaDeviceSynchronize();
	cudaEventDestroy(stop);
	cudaDeviceSynchronize();

	cudaMemcpy(h_finalAnswerFromGPU, d_array_data_output, dsize_histo, cudaMemcpyDeviceToHost);
	getAnyErrors("Copy Back\n");
	cudaDeviceSynchronize();

	printf("\nVerifying\n");
	verify(h_finalAnswerFromCPU, h_finalAnswerFromGPU, HISTOGRAM_SIZE);

	printf("\n----------------------------------------\n");
	printf("Execution Time for GPU: %.5f ms\n", secTotal);
	printf("Execution Time for CPU: %.5f ms\n", cpu_time);
	printf("Speedup : %.5f ms\n", cpu_time / secTotal);

	cudaFree(d_array_data_input);
	cudaFree(d_array_data_output);
	//Clean up memory
	//printf("dfdf11 \n");
	free(h_finalAnswerFromCPU);
	//printf("dfdf22 \n");
	free(h_finalAnswerFromGPU);
	//printf("dfdf33 \n");
	free(h_array_data_input_CPU);
	//printf("dfdf33ss \n");
}

