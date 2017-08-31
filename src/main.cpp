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

using namespace std;

//feedforward functions:
void augmentInput( const double *x, const int idx, double *xPrime ){
      xPrime[0] = 1.0;
      xPrime[DIMENSIONS] = x[idx];
}

void computeActivations( const double *xPrime, const double *W, double *a){
      //1. get activations: a_j = \sum_i^D{w_ji * x_i + w_j0}
      // perform matrix vector multiplication: a = W*x
      cout << "Computing 1st Layer Activations" << "\n";
      const double alpha = 1.0;
      const double beta = 0.0;
      const int incx = 1;
      cblas_dgemv( CblasRowMajor, CblasNoTrans, NUM_HIDDEN_NODES, (DIMENSIONS + 1),
		   alpha, W, (DIMENSIONS + 1), xPrime, incx, beta, a, incx);
}

int main(int argc, char *argv[])
{
      cout << " Aaron's Back." << endl;

      //--------------------------------------------------------------------------------
      // declare variables for calculations
      double *x, *t, *y, *X, *W, *V, *a, *z;

      x = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 ); // inputs
      t = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 ); // targets
      y = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 ); // outputs
      X = (double *)mkl_malloc( NUM_PATTERNS*(DIMENSIONS + 1)*sizeof( double ), 64 ); //data matrix
      W = (double *)mkl_malloc( NUM_HIDDEN_NODES * (DIMENSIONS + 1)*sizeof( double ), 64 ); //1st layer weights
      V = (double *)mkl_malloc( NUM_OUTPUTS*NUM_HIDDEN_NODES*sizeof( double ), 64 ); //2nd layer weights
      a = (double *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( double ), 64 ); // activations
      z = (double *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( double ), 64 ); // hidden nodes

      memset( x, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( t, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( y, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( X, 0.0,  NUM_PATTERNS *(DIMENSIONS + 1)* sizeof(double));
      memset( W, 0.0,  (DIMENSIONS + 1) * NUM_HIDDEN_NODES* sizeof(double));
      memset( V, 0.0,  NUM_OUTPUTS * NUM_HIDDEN_NODES  * sizeof(double));
      memset( a, 0.0,  NUM_HIDDEN_NODES * sizeof(double));
      memset( z, 0.0,  NUM_HIDDEN_NODES * sizeof(double));

      //--------------------------------------------------------------------------------
      //read data
      string inputsFile = "./data/xSquared/inputs.txt";
      string targetsFile = "./data/xSquared/targets.txt";

      loadData( x , inputsFile );
      loadData( t , targetsFile );
      
      cout << "Inputs" << endl;
      printVector( x, 10 );

      cout << "Targets" << endl;
      printVector( t, 10 );
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
      cout << "\nFirst layer weights" << endl;
      printMatrix( W,  NUM_HIDDEN_NODES, (DIMENSIONS + 1) );

      cout << "\nSecond layer weights" << endl;
      printMatrix( V, NUM_OUTPUTS, NUM_HIDDEN_NODES );
      //--------------------------------------------------------------------------------
      //4. compute activations:
      computeActivations( xPrime, W, a);
      cout << "\n\nActivations : " << endl;
      printVector( a, NUM_HIDDEN_NODES );
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
