#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include <mkl_dfti.h>

#define PI 3.141592653589793

int main() {
    /* Physical constants */
    const double omegaM = 4.5e11 * 2 * PI;    /* Modulation angular frequency */
    const double x0 = 1e-3;                   /* Base path length (m) */
    const double xm = 5e-8;                  /* Modulation amplitude (m) */

    /* Sampling parameters */
    const double dt = 1e-17;                  /* Time step (s) */
    const long long n = 4194304LL * 1024;     /* Total samples */

    printf("Allocating %lld samples...\n", n);

    /* Allocate aligned memory with MKL */
    double         *time_sig = (double*)mkl_malloc(n * sizeof(double), 64);
    MKL_Complex16  *spectrum = (MKL_Complex16*)mkl_malloc((n/2 + 1) * sizeof(MKL_Complex16), 64);

    if (!time_sig || !spectrum) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    /* Generate the time-domain signal in parallel */
    #pragma omp parallel for
    for (long long i = 0; i < n; ++i) {
        double t        = i * dt;
        double sin_term = sin(omegaM * t);
        time_sig[i]     = 1.0 / (x0 + xm * sin_term);
    }

    /* Create and configure MKL FFT descriptor */
    DFTI_DESCRIPTOR_HANDLE handle;
    MKL_LONG status;

    status = DftiCreateDescriptor(&handle,            /* descriptor handle   */
                                  DFTI_DOUBLE,        /* precision           */
                                  DFTI_REAL,          /* forward domain      */
                                  1,                  /* dimension           */
                                  n);                 /* length              */
    if (status) { fprintf(stderr, "DFTI create error %ld\n", status); return 1; }

    status = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (status) { fprintf(stderr, "DFTI set error %ld\n", status); return 1; }

    status = DftiCommitDescriptor(handle);
    if (status) { fprintf(stderr, "DFTI commit error %ld\n", status); return 1; }

    printf("Computing FFT with MKL...\n");
    status = DftiComputeForward(handle, time_sig, spectrum);
    if (status) { fprintf(stderr, "DFTI compute error %ld\n", status); return 1; }

    /* Write the frequency window around the modulation frequency */
    FILE *out = fopen("am_freqWindow.txt", "w");
    if (!out) { perror("fopen"); return 1; }

    const double f_center = omegaM / (2 * PI);
    const double f_half   = 5 * omegaM / (2 * PI);

    for (long long i = 0; i < n/2 + 1; ++i) { /* Only positive frequencies */
        double f   = i / (n * dt);
        double re  = spectrum[i].real;
        double im  = spectrum[i].imag;
        double amp = sqrt(re * re + im * im);
        if (f >= f_center - f_half && f <= f_center + f_half) {
            fprintf(out, "%.12e %.12e\n", f, amp);
        }
    }
    fclose(out);
    printf("Done. Output written to am_freqWindow.txt\n");

    /* Clean-up */
    DftiFreeDescriptor(&handle);
    mkl_free(time_sig);
    mkl_free(spectrum);
    return 0;
}
