#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>
#include "myheader.h"

////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////

cl_platform_id platform_id;         // platform id
cl_device_id device_id;             // compute device id 
cl_context context;                 // compute context
cl_command_queue commands;          // compute command queue
cl_program program;                 // compute program
cl_kernel kernel0;              // compute kernel

cl_int data[MAX_REC_N];
cl_int *h_input;
cl_int *h_output;

cl_mem d_input;
cl_mem d_output;

int
load_file_to_memory(const char *filename, char **result)
{ 
    int size = 0;
    FILE *f = fopen(filename, "rb");
    if (f == NULL) 
    { 
        *result = NULL;
        return -1; // -1 means file opening fail 
    } 
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *result = (char *)malloc(size+1);
    if (size != fread(*result, sizeof(char), size, f)) 
    { 
        free(*result);
        return -2; // -2 means file reading fail 
    } 
    fclose(f);
    (*result)[size] = 0;
    return size;
}

void init_device(){
    int  err;
    char cl_platform_vendor[1001];
    char cl_platform_name[1001];

    h_input  = (cl_int *) malloc(sizeof(cl_int)*MAX_REC_N);
    h_output = (cl_int *) malloc(sizeof(cl_int)*MAX_REC_N);

    err = clGetPlatformIDs(1,&platform_id,NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to find an OpenCL platform!\n");
        printf("Test failed\n");
        exit(1);
    }

    err = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, 1000, (void *)cl_platform_vendor,NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
        printf("Test failed\n");
        exit(1);
    }

    err = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME,1000,(void *)cl_platform_name,NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: clGetPlatformInfo(CL_PLATFORM_NAME) failed!\n");
        printf("Test failed\n");
        exit(1);
    }

    int fpga = 0;
#if defined (FPGA_DEVICE)
    fpga = 1;
#endif

    err = clGetDeviceIDs(platform_id, 
            fpga ? CL_DEVICE_TYPE_ACCELERATOR : CL_DEVICE_TYPE_CPU,
            1, 
            &device_id, 
            NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        printf("Test failed\n");
        exit(1);
    }


    // Create a compute context 
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        printf("Test failed\n");
        exit(1);
    }

    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        printf("Error: code %i\n",err);
        printf("Test failed\n");
        exit(1);
    }

    int status;

    unsigned char *kernelbinary;

    char xclbin[] = "pipe.xclbin";
    int n_i= load_file_to_memory(xclbin, (char **) &kernelbinary);
    if (n_i < 0) {
        printf("failed to load kernel from xclbin\n");
        printf("Test failed\n");
        exit(1);
    }

    size_t n = n_i;
    // Create the compute program from offline
    program = clCreateProgramWithBinary(context, 1, &device_id, &n,
            (const unsigned char **) &kernelbinary, &status, &err);
    if ((!program) || (err!=CL_SUCCESS)) {
        switch (err) {
            case CL_INVALID_CONTEXT:
                printf("CL_INVALID_CONTEXT\n");
                break;
            case CL_INVALID_VALUE:
                printf("CL_INVALID_VALUE\n");
                break;
            case CL_INVALID_DEVICE:
                printf("CL_INVALID_DEVICE\n");
                break;
            case CL_INVALID_BINARY:
                printf("CL_INVALID_BINARY\n");
                switch (status) {
                    case CL_SUCCESS:
                        printf("CL_SUCCESS\n");
                        break;
                    case CL_INVALID_VALUE:
                        printf("CL_INVALID_VALUE\n");
                        break;
                    case CL_INVALID_BINARY:
                        printf("CL_INVALID_BINARY\n");
                        break;
                }
                break;
            case CL_OUT_OF_HOST_MEMORY:
                printf("CL_OUT_OF_HOST_MEMORY\n");
                break;
        }
        printf("Error: Failed to create compute program from binary %d!\n", err);
        printf("Test failed\n");
        exit(1);
    }
    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        printf("Test failed\n");
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel0 = clCreateKernel(program, "kernel0", &err);
    if (!kernel0 || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel0! %d\n", err);
        printf("Test failed\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    d_input    = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(cl_int)*MAX_REC_N, NULL, NULL);
    d_output   = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int)*MAX_REC_N, NULL, NULL);

    if (!d_input || !d_output) {
        printf("Error: Failed to allocate device memory!\n");
        printf("Test failed\n");
        exit(1);
    }
    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel0, 0, sizeof(cl_mem), &d_input);
    err  = clSetKernelArg(kernel0, 1, sizeof(cl_mem), &d_output);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel argument d_input! %d\n", err);
        printf("Test failed\n");
        exit(1);
    }

}

void write_value(){

    for (int i = 0; i < MAX_REC_N; ++i) {
        h_input[i] = data[i];
    }
    printf("[host] inputs:\n");
    for (int i = 0; i < MAX_REC_N; ++i) {
        printf("%x ", h_input[i]);
    }
    printf("\n");
    int err;
    err = clEnqueueWriteBuffer(commands, d_input, CL_TRUE, 0, MAX_REC_N*sizeof(cl_int), h_input, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array input0!\n");
        printf("Test failed");
        exit(1);
    }

}

void execute_device(){
    int err;
#ifdef C_KERNEL
    err = clEnqueueTask(commands, kernel, 0, NULL, NULL);
#else
    size_t global[1];                   // global domain size for our calculation
    size_t local[1];                    // local domain size for our calculation
    global[0] = 1;
    local[0] = 1;

    err = clEnqueueNDRangeKernel(commands, kernel0, 1, NULL,
            (size_t*)global, (size_t*)local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel0! %d\n", err);
        printf("Test failed\n");
        exit(1);
    }

#endif
    clFinish(commands);
}

void read_value(){

    int err;
    cl_event readevent;
    err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0,
                              MAX_REC_N * sizeof(cl_int),
                              h_output, 0, NULL, &readevent);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        printf("Test failed\n");
        exit(1);
    }

    clWaitForEvents(1, &readevent);

    printf("\n[host] outputs:\n");
    for (int i = 0; i < MAX_REC_N; ++i) {
        printf("%x ", h_output[i]);
    }
    printf("\n");
}

int verify() {
    for (int i; i < MAX_REC_N; ++i) {
        if (h_input[i] != h_output[i]) {
            return 0;
        }
    }
    return 1;
}


int main(int argc, char** argv)
{
    for (int i = 0; i < MAX_REC_N; ++i) {
        data[i] = i;
    }
    init_device();
    write_value();
    execute_device();
    read_value();
    if (verify()) {
        printf("\nInput and output matched.\n");
    } else {
        printf("\nInput and output did not match.\n");
    }

    return 0;
}

