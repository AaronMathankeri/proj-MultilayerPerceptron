#include "backPropFunctions.hpp"

void computeOutputErrors( const double *y, const double *t, double *outputErrors){
      vdSub( NUM_OUTPUTS, y, &t[0], outputErrors);
}


void computeHiddenErrors( const double *a, const double *V, const double *outputErrors, double *inputErrors ){
      //1. get activations: d_j = h'(a)*\sum_k^D{v_kj * dk }
      const double alpha = 1.0;
      const double beta = 0.0;
      const int incx = 1;
      cblas_dgemv( CblasRowMajor, CblasTrans, NUM_OUTPUTS, NUM_HIDDEN_NODES,
		   alpha, V, NUM_HIDDEN_NODES, outputErrors, incx, beta, inputErrors, incx);

      //inputErrors * h'(aj)
      for (int j = 0; j < NUM_HIDDEN_NODES; ++j) {
	    inputErrors[j] *= dSigmoid( a[j] );
      }

}
