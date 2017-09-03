#include "helperFunctions.hpp"

double logisticSigmoid( const double a ){
      // sigma(a) = (1 + exp(-a) )^-1
      double z = 0.0;
      z = 1.0/( 1.0 + exp(-a) );
      return z;
}

double dSigmoid( const double a ){
      double x = 0.00;
      x = logisticSigmoid(a) * ( 1 - logisticSigmoid (a ) );
      return x;
}

double computeLeastSquaresError( const double *t, const double *y ){
      double error = 0.0;
      for (int i = 0; i < NUM_PATTERNS; ++i) {
	    error += (y[i] - t[i]) * (y[i] - t[i]);
      }
      error *= 0.5;
      return error;
}

