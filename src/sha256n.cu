// cd /home/hork/cuda-workspace/CudaSHA256/Debug/files
// time ~/Dropbox/FIIT/APS/Projekt/CpuSHA256/a.out -f ../file-list
// time ../CudaSHA256 -f ../file-list


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include "sha256n.cuh"
#include <ctype.h>

// Windows compatibility - remove unistd.h and dirent.h dependencies

char * trim(char *str){
    size_t len = 0;
    char *frontp = str;
    char *endp = NULL;

    if( str == NULL ) { return NULL; }
    if( str[0] == '\0' ) { return str; }

    len = strlen(str);
    endp = str + len;

    /* Move the front and back pointers to address the first non-whitespace
     * characters from each end.
     */
    while( isspace((unsigned char) *frontp) ) { ++frontp; }
    if( endp != frontp )
    {
        while( isspace((unsigned char) *(--endp)) && endp != frontp ) {}
    }

    if( str + len - 1 != endp )
            *(endp + 1) = '\0';
    else if( frontp != str &&  endp == frontp )
            *str = '\0';

    /* Shift the string so that it starts at str so that if it's dynamically
     * allocated, we can still free it on the returned pointer.  Note the reuse
     * of endp to mean the front of the string buffer now.
     */
    endp = str;
    if( frontp != str )
    {
            while( *frontp ) { *endp++ = *frontp++; }
            *endp = '\0';
    }


    return str;
}

__global__ void sha256_cuda(JOB ** jobs, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// perform sha256 calculation here
	if (i < n){
		SHA256_CTX ctx;
		sha256_init(&ctx);
		sha256_update(&ctx, jobs[i]->data, jobs[i]->size);
		sha256_final(&ctx, jobs[i]->digest);
	}
}

void pre_sha256() {
	// compy symbols
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}


void runJobs(JOB ** jobs, int n){
	int blockSize = 4;
	int numBlocks = (n + blockSize - 1) / blockSize;
	sha256_cuda <<< numBlocks, blockSize >>> (jobs, n);
}

// 🎯 GPU KERNEL: Optimized SHA-256 batch processing kernel
__global__ void sha256_batch_kernel(BYTE* input_data, size_t input_size, BYTE* output_data, int batch_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_count) return;

    // Calculate offset for this batch item
    BYTE* item_input = input_data + (idx * input_size);
    BYTE* item_output = output_data + (idx * 32);

    // SHA-256 context for this thread
    SHA256_CTX ctx;

    // Initialize SHA-256
    sha256_init(&ctx);

    // Process the input data
    sha256_update(&ctx, item_input, input_size);

    // Finalize and get hash
    sha256_final(&ctx, item_output);
}

// 🎯 PROPER GPU-OPTIMIZED SHA-256: Clean, fast, dedicated VRAM implementation
extern "C" void sha256n_batch_hash(BYTE* input_data, size_t input_size, BYTE* output_data, int batch_count) {
    //printf("DEBUG: Starting PROPER GPU SHA-256 batch processing for %d items\n", batch_count);

    // 🎯 STEP 1: Allocate flat GPU buffers (no complex structures)
    BYTE* d_input = nullptr;
    BYTE* d_output = nullptr;

    size_t total_input_size = batch_count * input_size;
    size_t total_output_size = batch_count * 32; // SHA-256 = 32 bytes

    //printf("DEBUG: Allocating GPU input buffer: %zu bytes\n", total_input_size);
    checkCudaErrors(cudaMalloc(&d_input, total_input_size));

    //printf("DEBUG: Allocating GPU output buffer: %zu bytes\n", total_output_size);
    checkCudaErrors(cudaMalloc(&d_output, total_output_size));

    // 🎯 STEP 2: Copy all input data to GPU in one transfer
    //printf("DEBUG: Copying all input data to GPU\n");
    checkCudaErrors(cudaMemcpy(d_input, input_data, total_input_size, cudaMemcpyHostToDevice));
    //printf("DEBUG: Input data copied to GPU successfully\n");

    // 🎯 STEP 3: Copy SHA-256 constants to GPU (if not already there)
    //printf("DEBUG: Ensuring SHA-256 constants are on GPU\n");
    pre_sha256();
    //printf("DEBUG: SHA-256 constants ready on GPU\n");

    // 🎯 STEP 4: Launch optimized GPU kernel
    //printf("DEBUG: Launching optimized SHA-256 kernel\n");
    // Calculate optimal grid/block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (batch_count + threadsPerBlock - 1) / threadsPerBlock;

    //printf("DEBUG: Kernel config - blocks: %d, threads: %d\n", blocksPerGrid, threadsPerBlock);

    // Launch the kernel with proper parameters
    sha256_batch_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, input_size, d_output, batch_count
    );

    // Check for kernel launch errors
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        printf("ERROR: Kernel launch failed: %s\n", cudaGetErrorString(kernelErr));
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }
    //printf("DEBUG: SHA-256 kernel launched successfully\n");

    // 🎯 STEP 5: Wait for completion and copy results
    //printf("DEBUG: Waiting for GPU kernel completion\n");
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("ERROR: GPU kernel synchronization failed: %s\n", cudaGetErrorString(syncErr));
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }
    //printf("DEBUG: GPU kernel completed successfully\n");

    // Copy results back
    //printf("DEBUG: Copying results back from GPU\n");
    checkCudaErrors(cudaMemcpy(output_data, d_output, total_output_size, cudaMemcpyDeviceToHost));
    //printf("DEBUG: All results copied successfully\n");

    // 🎯 STEP 6: Cleanup GPU memory
    //printf("DEBUG: Cleaning up GPU memory\n");
    cudaFree(d_input);
    cudaFree(d_output);
    //printf("DEBUG: GPU memory cleanup completed successfully\n");

    //printf("DEBUG: PROPER GPU SHA-256 batch processing completed successfully\n");
}


// 🎯 REMOVED: JOB_init function - no longer needed with new GPU-optimized approach


// 🎯 REMOVED: get_file_data function - no longer needed with new GPU-optimized approach

void print_usage(){
	printf("Usage: CudaSHA256 [OPTION] [FILE]...\n");
	printf("Calculate sha256 hash of given FILEs\n\n");
	printf("OPTIONS:\n");
	printf("\t-f FILE1 \tRead a list of files (separeted by \\n) from FILE1, output hash for each file\n");
	printf("\t-h       \tPrint this help\n");
	printf("\nIf no OPTIONS are supplied, then program reads the content of FILEs and outputs hash for each FILEs \n");
	printf("\nOutput format:\n");
	printf("Hash following by two spaces following by file name (same as sha256sum).\n");
	printf("\nNotes:\n");
	printf("Calculations are performed on GPU, each seperate file is hashed in its own thread\n");
}

// Main function removed for miner integration - we only need the batch SHA-256 function
