#include "myheader.h"

pipe int p0 __attribute__((xcl_reqd_pipe_depth(PIPE_LEN)));

kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void kernel0(__global int *g_input)
{
    __local int l_input[MAX_REC_N];

    async_work_group_copy(l_input, g_input, MAX_REC_N, 0);

    for (int i = 0; i < MAX_REC_N; ) {
        int ret = write_pipe(p0, &l_input[i]);
        if (ret == 0) {
            i++;
        }
    }

    return;
}

kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void kernel1(__global int *g_output)
{
    __local int l_output[MAX_REC_N];

    for (int i = 0; i < MAX_REC_N; ) {
        int ret = read_pipe(p0, &l_output[i]);
        if (ret == 0) {
            i++;
        }
    }

    async_work_group_copy(g_output, l_output, MAX_REC_N, 0);

    return;
}

