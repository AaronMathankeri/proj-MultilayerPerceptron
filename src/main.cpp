//Copyright (C) 2017 Aaron Mathankeri (aaron@quantifiedag.com)
//License: QuantifiedAG Software License.  See LICENSE.txt for full license.

/*
 *   \file example.cpp
 *   \brief A Documented file.
 *
 *  Detailed description
 *
 */
#include <iostream>
#include "mkl.h"
#include "ioFunctions.hpp"
#include "parameters.hpp"
#include "initializations.hpp"
#include "feedForwardFunctions.hpp"

using namespace std;

void computeOutputErrors( const double *y, const double *t, double *outputErrors){
      vdSub( NUM_OUTPUTS, y, &t[0], outputErrors);
}

double dSigmoid( const double a ){
      double x = 0.00;
      x = logisticSigmoid(a) * ( 1 - logisticSigmoid (a ) );
      cout << "derivative is " << x << endl;
      return x;
}
void computeHiddenErrors( const double *a, const double *V, const double *outputErrors, double *inputErrors ){
      //1. get activations: d_j = h'(a)*\sum_k^D{v_kj * dk }
      const double alpha = 1.0;
      const double beta = 0.0;
      const int incx = 1;
      cblas_dgemv( CblasRowMajor, CblasTrans, NUM_OUTPUTS, NUM_HIDDEN_NODES,
		   alpha, V, NUM_HIDDEN_NODES, outputErrors, incx, beta, inputErrors, incx);

      cout << "\n\nInput Errors are  : " << endl;
      printVector( inputErrors, NUM_HIDDEN_NODES );

      //inputErrors * h'(aj)
      for (int j = 0; j < NUM_HIDDEN_NODES; ++j) {
	    inputErrors[j] *= dSigmoid( a[j] );
      }

}

int main(int argc, char *argv[])
{
      cout << " Aaron's Back." << endl;

      //--------------------------------------------------------------------------------
      // declare variables for calculations
      double *x, *t, *y, *X, *W, *V, *a, *z;
      double *outputErrors, *inputErrors;

      x = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 ); // inputs
      t = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 ); // targets
      y = (double *)mkl_malloc( NUM_OUTPUTS*sizeof( double ), 64 ); // outputs
      X = (double *)mkl_malloc( NUM_PATTERNS*(DIMENSIONS + 1)*sizeof( double ), 64 ); //data matrix
      W = (double *)mkl_malloc( NUM_HIDDEN_NODES * (DIMENSIONS + 1)*sizeof( double ), 64 ); //1st layer weights
      V = (double *)mkl_malloc( NUM_OUTPUTS*NUM_HIDDEN_NODES*sizeof( double ), 64 ); //2nd layer weights
      a = (double *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( double ), 64 ); // activations
      z = (double *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( double ), 64 ); // hidden nodes

      outputErrors = (double *)mkl_malloc( NUM_OUTPUTS*sizeof( double ), 64 ); // activations
      inputErrors = (double *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( double ), 64 ); // hidden nodes

      memset( x, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( t, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( y, 0.0,  NUM_OUTPUTS * sizeof(double));
      memset( X, 0.0,  NUM_PATTERNS *(DIMENSIONS + 1)* sizeof(double));
      memset( W, 0.0,  (DIMENSIONS + 1) * NUM_HIDDEN_NODES* sizeof(double));
      memset( V, 0.0,  NUM_OUTPUTS * NUM_HIDDEN_NODES  * sizeof(double));
      memset( a, 0.0,  NUM_HIDDEN_NODES * sizeof(double));
      memset( z, 0.0,  NUM_HIDDEN_NODES * sizeof(double));

      memset( outputErrors, 0.0,  NUM_OUTPUTS * sizeof(double));
      memset( inputErrors, 0.0,  NUM_HIDDEN_NODES * sizeof(double));
      //--------------------------------------------------------------------------------
      //read data
      string inputsFile = "./data/xSquared/inputs.txt";
      string targetsFile = "./data/xSquared/targets.txt";

      loadData( x , inputsFile );
      loadData( t , targetsFile );
      /*
      cout << "Inputs" << endl;
      printVector( x, 10 );

      cout << "Targets" << endl;
      printVector( t, 10 );
      */
      //--------------------------------------------------------------------------------
      //2. each input pattern must have a x0 clamped at 1.0 for the bias
      double *xPrime = (double *)mkl_malloc( (DIMENSIONS + 1)*sizeof( double ), 64 ); // inputs
      //for (int i = 0; i < NUM_PATTERNS; ++i) {
      for (int i = 0; i < 1; ++i) {
	    memset( xPrime, 0.0, (DIMENSIONS + 1)* sizeof(double));
	    augmentInput( x, i, xPrime );
	    //cout << "Xprime is " << xPrime[0] << "\t" <<xPrime[1] << endl;
      }
      cout << "Xprime is " << xPrime[0] << "\t" <<xPrime[1] << endl;
      //computeDataMatrix( x, X );
      //cout << "\nData matrix" << endl;
      //printMatrix( X, NUM_PATTERNS, 2 );
      //--------------------------------------------------------------------------------
      //3. Randomly initialize weights
      setRandomWeights( W,  NUM_HIDDEN_NODES, (DIMENSIONS+1) );
      setRandomWeights( V, NUM_OUTPUTS, NUM_HIDDEN_NODES );
      //*
      cout << "\nFirst layer weights" << endl;
      printMatrix( W,  NUM_HIDDEN_NODES, (DIMENSIONS + 1) );

      cout << "\nSecond layer weights" << endl;
      printMatrix( V, NUM_OUTPUTS, NUM_HIDDEN_NODES );
      //*/
      //--------------------------------------------------------------------------------
      //FEEDFORWARD FUNCTIONS!!!!
      cout << "\n\nFORWARD PROPAGATE THROUGH NETWORK " << endl;
      //4. compute activations:
      computeActivations( xPrime, W, a);
      cout << "\n\nActivations : " << endl;
      printVector( a, NUM_HIDDEN_NODES );
      //--------------------------------------------------------------------------------
      //5. compute hidden nodes
      computeHiddenUnits( a, z);
      //cout << "\n\nHidden Nodes : " << endl;
      //printVector( z, NUM_HIDDEN_NODES );
      //--------------------------------------------------------------------------------
      //6. compute output
      computeOutputActivations( z , V, y );
      cout << "\n\nOutput is  : " << endl;
      printVector( y, NUM_OUTPUTS );
      //--------------------------------------------------------------------------------
      // ERROR BACKPROPAGATION
      //1. computeOutputErrors
      cout << "\n\nBACKPROPAGATE THROUGH NETWORK " << endl;
      computeOutputErrors( y, t, outputErrors );
      cout << "\n\nOutput Errors are  : " << endl;
      printVector( outputErrors, NUM_OUTPUTS );
      //--------------------------------------------------------------------------------
      computeHiddenErrors( a, V, outputErrors, inputErrors );
      cout << "\n\nInput Errors are  : " << endl;
      printVector( inputErrors, NUM_HIDDEN_NODES );
      //--------------------------------------------------------------------------------
      //--------------------------------------------------------------------------------
      //--------------------------------------------------------------------------------
      //--------------------------------------------------------------------------------

      printf ("\n Deallocating memory \n\n");
      mkl_free( x );
      mkl_free( t );
      mkl_free( y );
      mkl_free( X );
      mkl_free( W );
      mkl_free( V );
      mkl_free( a );
      mkl_free( z );
      printf (" Example completed. \n\n");

      return 0;
}
