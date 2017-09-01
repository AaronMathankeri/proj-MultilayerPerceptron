#include "feedForwardFunctions.hpp"

//---------------------------------------------------------------------------------
//feedforward functions:
void computeActivations( const double *xPrime, const double *W, double *a){
      //1. get activations: a_j = \sum_i^D{w_ji * x_i + w_j0}
      // perform matrix vector multiplication: a = W*x
      const double alpha = 1.0;
      const double beta = 0.0;
      const int incx = 1;
      cblas_dgemv( CblasRowMajor, CblasNoTrans, NUM_HIDDEN_NODES, (DIMENSIONS + 1),
		   alpha, W, (DIMENSIONS + 1), xPrime, incx, beta, a, incx);
}
//---------------------------------------------------------------------------------
void computeHiddenUnits( const double *a, double* z){
      z[0] = 1.0;
      for (int j = 0; j < NUM_HIDDEN_NODES; ++j) {
	    z[j+1] = logisticSigmoid( a[j] );
      }
}
//---------------------------------------------------------------------------------
void computeOutputActivations( const double *z, const double *V, double *y){
      // perform matrix vector multiplication: y = V*z
      const double alpha = 1.0;
      const double beta = 0.0;
      const int incx = 1;
      cblas_dgemv( CblasRowMajor, CblasNoTrans, NUM_OUTPUTS, (NUM_HIDDEN_NODES+1),
		   alpha, V, (NUM_HIDDEN_NODES+1), z, incx, beta, y, incx);
}
//---------------------------------------------------------------------------------

