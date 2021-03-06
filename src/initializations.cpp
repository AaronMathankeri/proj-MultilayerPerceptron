#include "initializations.hpp"

void computeDataMatrix( const double *x, double *X ){
      double *ones;
      const int incx = 1;
      
      ones = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );
      fill_n(ones, NUM_PATTERNS, 1.0); // create ones vector

      //set first column to 1--dummy index to calculate w0
      cblas_dcopy(NUM_PATTERNS, ones, incx, X, (DIMENSIONS+1));

      // set columns 1 ... M-1 with basis function vectors
      for (int i = 0; i < NUM_PATTERNS; ++i) {
	    for (int j = 0; j < (DIMENSIONS+1); ++j) {
		  if ( j > 0) {
			int p = j - 1;
			X[i*(DIMENSIONS+1) + j] = x[i*(DIMENSIONS) + p];			
		  }
	    }
      }
      mkl_free( ones );
}

double fRand(double fMin, double fMax){
      double f = (double)rand() / RAND_MAX;
      return fMin + f * (fMax - fMin);
}

void setRandomWeights( double * weights, int nRows, int nCols ){
      for (int i = 0; i < (nRows*nCols); ++i) {
	    double temp = fRand( -5.0, 10.0);
	    weights[i] = temp;
      }
}

void augmentInput( const double *x, const int idx, double *xPrime ){
      xPrime[0] = 1.0;
      xPrime[DIMENSIONS] = x[idx];
}
