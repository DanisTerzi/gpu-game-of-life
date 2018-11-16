#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <helper_cuda.h>
#include <helper_timer.h>
#include "GameOfLife.h"
#include "Methods.h"

void usage(){

   fprintf(stderr, "usage: ./life gameGridWidth gameGridHeight blockWidth blockHeight numOfPasses\n");
   fprintf(stderr, "usage: ./life\n");
}

int main (int argc, const char * argv[]) {
	// Create a grid (2d array of values) for Game of Life
	int gameGridHeight, gameGridWidth, blockHeight, blockWidth, numOfPasses;
	if (argc != 1 || argc != 6)
		usage();
	if (argc == 6){
		gameGridWidth = strToInt((char *)argv[1]);
		gameGridHeight = strToInt((char *)argv[2]);
		blockWidth = strToInt((char *)argv[3]);
		blockHeight = strToInt((char *)argv[4]);
		numOfPasses = strToInt((char *)argv[5]);
	}
	else{
		gameGridWidth = 32;
		gameGridHeight = 10;
		blockWidth = 32;
		blockHeight = 10;
		numOfPasses = 18;
	}
	int *gameGridIn = (int *) malloc(gameGridHeight * gameGridWidth * sizeof(int));
	bool pause = false;
	// Initialize the grid
	initializeArrays(gameGridIn, gameGridWidth, gameGridHeight);
	
	size_t pitch;

	// allocate device memory for data in
	int *d_gameGridIn;
	cudaMallocPitch( (void**) &d_gameGridIn, &pitch, gameGridWidth * sizeof(int), gameGridHeight);
	
	// copy host memory to device memory for data in
	cudaMemcpy2D( d_gameGridIn, pitch, gameGridIn, gameGridWidth * sizeof(int), gameGridWidth * sizeof(int), gameGridHeight, cudaMemcpyHostToDevice);

	int gridWidth = (int) ceil( (gameGridWidth) / (float)blockWidth);
	int gridHeight = (int) ceil( (gameGridHeight) / (float)blockHeight);
	printf("block width: %d, block height: %d, grid width: %d, grid height: %d,\n\n", blockWidth, blockHeight, gridWidth, gridHeight);

	// Each block gets a shared memory region of this size.
	unsigned int shared_mem_size = ((blockWidth + 2) * (blockHeight+2)) * sizeof(int); 

	// Format the grid, which is a collection of blocks. 
   	dim3  grid( gridWidth, gridHeight, 1);
   
   	// Format the blocks. 
   	dim3  threads( blockWidth, blockHeight, 1);

	// When game is paused - allow the user to modify the grid values
	// When game is played - make the grid follow the rules
	printf("Starting grid:\n");	
	printArray(gameGridIn, gameGridHeight, gameGridWidth, 1);
	for (int i = 0; (i < numOfPasses) && (!pause); i++){
		//execute the kernel
		playGame<<< grid, threads, shared_mem_size >>>( d_gameGridIn, pitch/sizeof(int), gameGridWidth, gameGridHeight);
		//Print the array
		cudaMemcpy2D( gameGridIn, gameGridWidth * sizeof(int), d_gameGridIn, pitch, gameGridWidth * sizeof(int), gameGridHeight, cudaMemcpyDeviceToHost);
		printf("Grid Generation: %d\n", i+1);
		printArray(gameGridIn, gameGridHeight, gameGridWidth, 1);
	}

	cudaThreadSynchronize();
	
	cudaMemcpy2D( gameGridIn, gameGridWidth * sizeof(int), d_gameGridIn, pitch, gameGridWidth * sizeof(int), gameGridHeight,cudaMemcpyDeviceToHost);

}
