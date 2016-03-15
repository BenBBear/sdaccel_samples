#include "myheader.h"

pipe int p0 __attribute__((xcl_reqd_pipe_depth(PIPE_DEPTH)));
pipe int p1 __attribute__((xcl_reqd_pipe_depth(PIPE_DEPTH)));

kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void kernel_in(__global int *g_input)
{
    __local int l_input[REC_N];

    async_work_group_copy(l_input, g_input, REC_N, 0);

    for (int i = 0; i < REC_N; ) {
        int ret = write_pipe(p0, &l_input[i]);
        if (ret == 0) {// write_pipe successful
            ++i;
        }
    }

    return;
}

kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void kernel_inter(void)
{
    int data;

    for (int i = 0; i < REC_N; ) {
        int ret = read_pipe(p0, &data);
        if (ret == 0) {
            ++data;
            while (write_pipe(p1, &data) != 0) {}
            ++i;
        }
    }

    return;
}

kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void kernel_out(__global int *g_output)
{
    __local int l_output[REC_N];

    for (int i = 0; i < REC_N; ) {
        int ret = read_pipe(p1, &l_output[i]);
        if (ret == 0) {
            ++i;
        }
    }

    async_work_group_copy(g_output, l_output, REC_N, 0);

    return;
}

