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

void computeGradV( const double *outputErrors, const double *z, double *gradV ){
      //dE/dV = dk * z
      for (int k = 0; k < NUM_OUTPUTS; ++k) {
	    for (int j = 0; j < NUM_HIDDEN_NODES; ++j) {
		  gradV[k*NUM_HIDDEN_NODES + j] = outputErrors[k] * z[j]; 		  
	    }
      }

}

void computeGradW( const double *inputErrors, const double *x, double *gradW ){
      for (int j = 0; j < NUM_HIDDEN_NODES; ++j) {
	    for (int i = 0; i < (DIMENSIONS+1); ++i) {
		  gradW[j*(DIMENSIONS+1) + i] = inputErrors[j] * x[i]; 		  
	    }
      }
}
