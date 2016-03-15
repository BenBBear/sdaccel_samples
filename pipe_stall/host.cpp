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
cl_kernel kernel_in;                // compute kernel
cl_kernel kernel_inter;            // compute kernel
cl_kernel kernel_out;               // compute kernel

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

void init_device(int concurrent){
    int  err;
    char cl_platform_vendor[1001];
    char cl_platform_name[1001];

    h_input  = (cl_int *) malloc(sizeof(cl_int)*REC_N);
    h_output = (cl_int *) malloc(sizeof(cl_int)*REC_N);

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
    if (concurrent) {
        commands = clCreateCommandQueue(context, device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    } else {
        commands = clCreateCommandQueue(context, device_id, 0, &err);
    }
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
    kernel_in = clCreateKernel(program, "kernel_in", &err);
    if (!kernel_in || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel_in! %d\n", err);
        printf("Test failed\n");
        exit(1);
    }
    kernel_inter = clCreateKernel(program, "kernel_inter", &err);
    if (!kernel_inter || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel_inter! %d\n", err);
        printf("Test failed\n");
        exit(1);
    }
    kernel_out = clCreateKernel(program, "kernel_out", &err);
    if (!kernel_out || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel_out! %d\n", err);
        printf("Test failed\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    d_input    = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(cl_int)*REC_N, NULL, NULL);
    d_output   = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int)*REC_N, NULL, NULL);

    if (!d_input || !d_output) {
        printf("Error: Failed to allocate device memory!\n");
        printf("Test failed\n");
        exit(1);
    }
    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel_in, 0, sizeof(cl_mem), &d_input);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel argument d_input! %d\n", err);
        printf("Test failed\n");
        exit(1);
    }

    err = 0;
    err  = clSetKernelArg(kernel_out, 0, sizeof(cl_mem), &d_output);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel argument d_output! %d\n", err);
        printf("Test failed\n");
        exit(1);
    }
}

void write_value(){

    for (int i = 0; i < REC_N; ++i) {
        h_input[i] = i;
    }
    printf("[host] inputs:\n");
    for (int i = 0; i < REC_N; ++i) {
        printf("%d ", h_input[i]);
    }
    printf("\n");
    int err;
    err = clEnqueueWriteBuffer(commands, d_input, CL_TRUE, 0, REC_N*sizeof(cl_int), h_input, 0, NULL, NULL);
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

    err = clEnqueueNDRangeKernel(commands, kernel_in, 1, NULL,
            (size_t*)global, (size_t*)local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel_in! %d\n", err);
        printf("Test failed\n");
        exit(1);
    }

    err = clEnqueueNDRangeKernel(commands, kernel_inter, 1, NULL,
            (size_t*)global, (size_t*)local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel_inter! %d\n", err);
        printf("Test failed\n");
        exit(1);
    }

    err = clEnqueueNDRangeKernel(commands, kernel_out, 1, NULL,
            (size_t*)global, (size_t*)local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel_out! %d\n", err);
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
                              REC_N * sizeof(cl_int),
                              h_output, 0, NULL, &readevent);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        printf("Test failed\n");
        exit(1);
    }

    clWaitForEvents(1, &readevent);

    printf("\n[host] outputs:\n");
    for (int i = 0; i < REC_N; ++i) {
        printf("%d ", h_output[i]);
    }
    printf("\n");
}

int main(int argc, char** argv)
{
    int concurrent;
    if (argc != 2) {
        puts("Give 1 for concurrent OpenCL kernel execution and 0 otherwise.");
        exit(0);
    }
    concurrent = atoi(argv[1]);
    init_device(concurrent);
    write_value();
    execute_device();
    read_value();

    return 0;
}

