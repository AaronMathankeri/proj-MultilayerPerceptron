#include "gradientDescent.hpp"

void applyLearningRate( double *x, const int nElements, const double learningRate ){
      const int incx = 1;
      double *y = (double *)mkl_malloc( nElements*sizeof( double ), 64 ); // inputs
      std::memset( y, 0.0,  nElements * sizeof(double));

      cblas_daxpy( nElements, learningRate, x, incx, y, incx );
      cblas_dcopy( nElements, y, incx, x, incx );
      
      mkl_free( y );
}

void updateWeights( double *weights, const double *deltaWeights, const int nElements ){
      vdSub( nElements, weights, deltaWeights, weights);
}
