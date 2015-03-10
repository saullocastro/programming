#include <cblas.h>
#include <stdlib.h>
 
int main()
{
  int m, n, k, i, j;
  double *A, *B, *C;
  double alpha, beta;
  m = 300;
  k = 1000;
  n = 2000;
  A = (double *) malloc (m * k * sizeof(double));
  B = (double *) malloc (k * n * sizeof(double));
  C = (double *) malloc (m * n * sizeof(double));
  alpha = 1.;
  beta = 0.;
  for (i=0 ; i<m ; i++) {
    for (j=0 ; j<k ; j++) {
      A[i,j] = 0.1*random(10);
    };
  };
  for (i=0 ; i<k ; i++) {
    for (j=0 ; j<n ; j++) {
      B[i,j] = 0.1*random(5);
    };
  };
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              m, n, k, alpha, A, k, B, n, beta, C, n);
  printf("Product of entered matrices:-\n", C);
  for(i=0; i < m; i++) {
    for(j=0; j < n; j++) {
      printf("%d ", C[i,j]);
    };
    printf("\n");
  };

  free(A);
  free(B);
  free(C);
  return 0;
}
