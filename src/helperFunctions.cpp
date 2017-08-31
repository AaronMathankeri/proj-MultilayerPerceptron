#include "helperFunctions.hpp"

double logisticSigmoid( const double a ){
      // sigma(a) = (1 + exp(-a) )^-1
      double z = 0.0;
      z = 1.0/( 1.0 + exp(-a) );
      return z;
}
