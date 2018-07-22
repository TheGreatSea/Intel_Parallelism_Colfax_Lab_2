#include <mkl.h>
#include "distribution.h"

/* 
What I mainly did is same as written: generate random numbers, loop-interchange, forcible vectorization (#pragma omp simd), store particle position, check counts outside nested loops. In my case, vector dependency over nested loops was still remained, so it was partially non-vectorized.
*/

//vectorize this function based on instruction on the lab page
int diffusion(const int n_particles, 
              const int n_steps, 
              const float x_threshold,
              const float alpha, 
              VSLStreamStatePtr rnStream) {
  int n_escaped=0;

       float rn[n_particles];
      
      //Intel MKL function to generate random numbers
      vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, n_particles, rn, -1.0, 1.0);
       
       float x[n_particles];
       
       for (int j = 0; j < n_steps; j++) {
         #pragma omp simd
           for (int i = 0; i < n_particles; i++) {
             x[i] = 0.0f;
             x[i] += dist_func(alpha, rn);
             } // for {i}
    
  } // for {j}
  for (int k = 0; k < n_particles; k++) {
    if (x > x_threshold) n_escaped++;
    } // for {k}
    return n_escaped;
} // diffusion