#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
//#include <helper_functions.h>
//#include <helper_cuda.h>

#include <R.h>
#include <Rinternals.h>

/* Lil test script
 * A <- matrix(c(1,2,3,4,5,6), ncol=2)
 * B <- matrix(c(1,2,3,4,5,6), ncol=3)
 * C <- A %*% B
 * C_gpu=gpumult(A,B)
 */

extern "C" SEXP GPUMULT( SEXP A_,SEXP B_)
{
  double * A = REAL(A_);
  double * B = REAL(B_);
  int m = Rf_nrows(A_);
  int k = Rf_ncols(A_);
  int k_test = Rf_nrows(B_);
  int n = Rf_ncols(B_);
  if (k != k_test)
  {
    printf("bad matrix dimensions %d,%d,%d\n",m,k,n);
    return NULL;
  }
  cublasHandle_t cublas_handle;
  cublasStatus_t status;
  cudaError_t cudastatus;
  if( CUBLAS_STATUS_SUCCESS != (status = cublasCreate(&cublas_handle))) {
    printf("cublasCreate: %s\n", cublasGetStatusString(status));
    return NULL;
  }
  SEXP result_;
  PROTECT(result_ = allocMatrix(REALSXP, m, n));
  double * result = REAL(result_);
  printf("version %d,%d,%d v12\n",m,k,n);
  // Do the GPU stuff
  // Allocate memory space on the device 
  double *d_a, *d_b, *d_c;
  if (cudaSuccess != (cudastatus = cudaMalloc((void **) &d_a, sizeof(double)*m*k))) {
    printf("malloc d_a: %s\n", cudaGetErrorName(cudaGetLastError()));
    return NULL;
  }
  if (cudaSuccess != (cudastatus = cudaMalloc((void **) &d_b, sizeof(double)*k*n))) {
    printf("malloc d_b: %s\n", cudaGetErrorName(cudaGetLastError()));
    return NULL;
  }
  if (cudaSuccess != (cudastatus = cudaMalloc((void **) &d_c, sizeof(double)*m*n))) {
    printf("malloc d_c: %s\n", cudaGetErrorName(cudaGetLastError()));
    return NULL;
  }
  //printf("A[0]=%3.2f  B[1]=%3.2f",A[0],B[0]);
  // copy matrix A and B from host to device memory
  if (cudaSuccess != (cudastatus = cudaMemcpy(d_a, A,  sizeof(double)*m*k, cudaMemcpyHostToDevice))) {
    printf("memcpy A->d_a: %s\n", cudaGetErrorName(cudaGetLastError()));
    return NULL;
  }
  if (cudaSuccess != (cudastatus = cudaMemcpy(d_b, B, sizeof(double)*k*n, cudaMemcpyHostToDevice))) {
    printf("memcpy B->d_b: %s\n", cudaGetErrorName(cudaGetLastError()));
    return NULL;
  }

  // Note CUBLAS is column major as is R so do not transpose input order
  const double alpha = 1;
  const double beta = 0;
  if( CUBLAS_STATUS_SUCCESS != (status = cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    &alpha,
    d_a, m,
    d_b, k,
    &beta,
    d_c, m
  ))) {
    printf("cublasDgemm: %s\n", cublasGetStatusString(status));
    return NULL;
  }
  
  // No copy back
  if (cudaSuccess != (cudastatus = cudaMemcpy(result, d_c, sizeof(double)*m*n, cudaMemcpyDeviceToHost))) {
    printf("memcpy d_c->C: %s\n", cudaGetErrorName(cudaGetLastError()));
    return NULL;
  }
  //printf("r[0,0]=%3.2f\n",result[0]);
  // free memory on device  (memory on host is managed by R)
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  if( CUBLAS_STATUS_SUCCESS != (status = cublasDestroy(cublas_handle))) {
    printf("cublasDestroy: %s\n", cublasGetStatusString(status));
    return NULL;
  }
  
  UNPROTECT(1);
  return result_;
}
