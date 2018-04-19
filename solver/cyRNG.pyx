# cython: profile=False

"""
TAKEN FROM:
https://studywolf.wordpress.com/tag/cython/

based on:
https://www.johndcook.com/blog/cpp_random_number_generation/
"""
cdef extern from 'SimpleRNG.h':
    cdef cppclass SimpleRNG:
        SimpleRNG()
        # ap

        # Seed the random number generator
        void SetState(unsigned int u, unsigned int v)

        # A uniform random sample from the open interval (0, 1)
        double GetUniform()

        # A uniform random sample from the set of unsigned integers
        unsigned int GetUint()

        # Normal (Gaussian) random sample
        double GetNormal(double mean, double standardDeviation)

        # Exponential random sample
        double GetExponential(double mean)

        # Gamma random sample
        double GetGamma(double shape, double scale)

        # Chi-square sample
        double GetChiSquare(double degreesOfFreedom)

        # Inverse-gamma sample
        double GetInverseGamma(double shape, double scale)

        # Weibull sample
        double GetWeibull(double shape, double scale)

        # Cauchy sample
        double GetCauchy(double median, double scale)

        # Student-t sample
        double GetStudentT(double degreesOfFreedom)

        # The Laplace distribution is also known as the double exponential distribution.
        double GetLaplace(double mean, double scale)

        # Log-normal sample
        double GetLogNormal(double mu, double sigma)

        # Beta sample
        double GetBeta(double a, double b)

        # Poisson sample
        int GetPoisson(double lam)


cdef class cyRNG:
    cdef SimpleRNG* thisptr # hold a C++ instance
    def __cinit__(self):
        self.thisptr = new SimpleRNG()
    def __dealloc__(self):
        del self.thisptr

    # Seed the random number generator
    cpdef void SetState(self, unsigned int u, unsigned int v):
        self.thisptr.SetState(u, v)

    # A uniform random sample from the open interval (0, 1)
    cpdef double GetUniform(self):
        return self.thisptr.GetUniform()

    # A uniform random sample from the set of unsigned integers
    cpdef unsigned int GetUint(self):
        return self.thisptr.GetUint()

    # Normal (Gaussian) random sample
    cpdef double GetNormal(self, double mean, double std_dev):
        return self.thisptr.GetNormal(mean, std_dev)

    # Exponential random sample
    cpdef double GetExponential(self, double mean):
        return self.thisptr.GetExponential(mean)

    # Gamma random sample
    cpdef double GetGamma(self, double shape, double scale):
        return self.thisptr.GetGamma(shape, scale)

    # Chi-square sample
    cpdef double GetChiSquare(self, double degreesOfFreedom):
        return self.thisptr.GetChiSquare(degreesOfFreedom)

    # Inverse-gamma sample
    cpdef double GetInverseGamma(self, double shape, double scale):
        return self.thisptr.GetInverseGamma(shape, scale)

    # Weibull sample
    cpdef double GetWeibull(self, double shape, double scale):
        return self.thisptr.GetWeibull(shape, scale)

    # Cauchy sample
    cpdef double GetCauchy(self, double median, double scale):
        return self.thisptr.GetCauchy(median, scale)

    # Student-t sample
    cpdef double GetStudentT(self, double degreesOfFreedom):
        return self.thisptr.GetStudentT(degreesOfFreedom)

    # The Laplace distribution is also known as the double exponential distribution.
    cpdef double GetLaplace(self, double mean, double scale):
        return self.thisptr.GetLaplace(mean, scale)

    # Log-normal sample
    cpdef double GetLogNormal(self, double mu, double sigma):
        return self.thisptr.GetLogNormal(mu, sigma)

    # Beta sample
    cpdef double GetBeta(self, double a, double b):
        return self.thisptr.GetBeta(a, b)

    # Poisson sample
    cpdef int GetPoisson(self, double lam):
        return self.thisptr.GetPoisson(lam)
