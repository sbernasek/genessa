from libc.stdlib cimport rand, srand, RAND_MAX

cpdef list generate_floats(int seed, int N):
    srand(seed)
    return [rand()/(RAND_MAX*1.0) for _ in range(N)]
