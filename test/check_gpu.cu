/*
source code for checking which GPU you are running the code on.

load the cuda module:
$ module load cuda11/11.0

to compile: 
$ nvcc -o check check_gpu.cu

*/


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

int main(int argc, char **argv)
{


    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;


        if (cudaGetDeviceProperties(&devProps, dev) == 0)
        {
           printf("Using device %d:\n", dev);
           printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem, 
               (int)devProps.major, (int)devProps.minor, 
               (int)devProps.clockRate);
        }
        
    
        
    return 0;
}
