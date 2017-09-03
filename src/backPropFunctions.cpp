#include "backPropFunctions.hpp"

void computeOutputErrors( const double *y, const double *t, double *outputErrors){
      vdSub( NUM_OUTPUTS, y, t, outputErrors);
}


void computeHiddenErrors( const double *a, const double *W, const double *outputErrors, double *inputErrors ){
      //1. get activations: d_j = h'(a)*\sum_k^D{v_kj * dk }
      /*
      const double alpha = 1.0;
      const double beta = 0.0;
      const int incx = 1;
      cblas_dgemv( CblasRowMajor, CblasTrans, NUM_OUTPUTS, NUM_HIDDEN_NODES+1,
		   alpha, V, NUM_HIDDEN_NODES+1, outputErrors, incx, beta, inputErrors, incx);
      */
      for (int j = 0; j < NUM_HIDDEN_NODES; ++j) {
	    for (int k = 0; k < NUM_OUTPUTS; ++k) {
		  inputErrors[j] += W[j+1] * outputErrors[k];
	    }
      }
      //inputErrors * h'(aj)
      for (int j = 0; j < NUM_HIDDEN_NODES; ++j) {
	    inputErrors[j] *= dSigmoid( a[j] );
      }
}

void computeGradW( const double *outputErrors, const double *z, double *gradW ){
      //dE/dV = dk * z
      for (int k = 0; k < NUM_OUTPUTS; ++k) {
	    for (int j = 0; j < NUM_HIDDEN_NODES+1; ++j) {
		  gradW[k*(NUM_HIDDEN_NODES+1) + j] = outputErrors[k] * z[j]; 		  
	    }
      }

}

void computeGradV( const double *inputErrors, const double *x, double *gradV ){
      for (int j = 0; j < NUM_HIDDEN_NODES; ++j) {
	    for (int i = 0; i < (DIMENSIONS+1); ++i) {
		  gradV[j*(DIMENSIONS+1) + i] = inputErrors[j] * x[i]; 		  
	    }
      }
}
