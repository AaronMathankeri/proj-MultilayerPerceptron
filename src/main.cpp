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
#include "backPropFunctions.hpp"

using namespace std;


void applyLearningRate( double *x, const int nElements, const double learningRate ){
      const int incx = 1;
      double *y = (double *)mkl_malloc( nElements*sizeof( double ), 64 ); // inputs
      memset( y, 0.0,  nElements * sizeof(double));
      cblas_daxpy( nElements, learningRate, x, incx, y, incx );
      cblas_dcopy( nElements, y, incx, x, incx );
      
      mkl_free( y );
}

void updateWeights( double *weights, const double *deltaWeights, const int nElements ){
      vdSub( nElements, weights, deltaWeights, weights);
}


int main(int argc, char *argv[])
{
      cout << " Aaron's Back." << endl;

      //--------------------------------------------------------------------------------
      // declare variables for calculations
      double *x, *t, *y, *X, *W, *V, *a, *z;
      double *outputErrors, *inputErrors;
      double *gradW, *gradV;

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

      gradW = (double *)mkl_malloc( NUM_HIDDEN_NODES * (DIMENSIONS + 1)*sizeof( double ), 64 ); //1st layer weights
      gradV = (double *)mkl_malloc( NUM_OUTPUTS*NUM_HIDDEN_NODES*sizeof( double ), 64 ); //2nd layer weights
      
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
      memset( gradW, 0.0,  (DIMENSIONS + 1) * NUM_HIDDEN_NODES* sizeof(double));
      memset( gradV, 0.0,  NUM_OUTPUTS * NUM_HIDDEN_NODES  * sizeof(double));
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
      computeGradV( outputErrors, z , gradV );
      computeGradW( inputErrors, x, gradW );
      cout << "\n\n GradV   : " << endl;
      printMatrix( gradV, NUM_OUTPUTS ,NUM_HIDDEN_NODES );
      cout << "\n\n GradW  : " << endl;
      printMatrix( gradW, NUM_HIDDEN_NODES, (DIMENSIONS+1) );
      //--------------------------------------------------------------------------------
      // STOCHASTIC GRADIENT DESCENT
      //1. get gradient
      //2. multiply by learning rate
      //3. subtract current weights
      cout << "\n\nSTOCHASTIC GRADIENT DESCENT " << endl;
      const double learningRate = 1e-4;
      //--------------------------------------------------------------------------------
      //multiply weights by learningrate
      applyLearningRate( gradV, (NUM_OUTPUTS*NUM_HIDDEN_NODES), learningRate );
      applyLearningRate( gradW, (NUM_HIDDEN_NODES*(DIMENSIONS+1)), learningRate );
      cout << "\n\n GradV   : " << endl;
      printMatrix( gradV, NUM_OUTPUTS ,NUM_HIDDEN_NODES );
      cout << "\n\n GradW  : " << endl;
      printMatrix( gradW, NUM_HIDDEN_NODES, (DIMENSIONS+1) );
      //--------------------------------------------------------------------------------
      updateWeights( W, gradW, (NUM_HIDDEN_NODES*(DIMENSIONS+1)));
      updateWeights( V, gradV, (NUM_OUTPUTS*NUM_HIDDEN_NODES) );
      cout << "NEW Weights are.." << endl;
      cout << "\nFirst layer weights" << endl;
      printMatrix( W,  NUM_HIDDEN_NODES, (DIMENSIONS + 1) );

      cout << "\nSecond layer weights" << endl;
      printMatrix( V, NUM_OUTPUTS, NUM_HIDDEN_NODES );
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
