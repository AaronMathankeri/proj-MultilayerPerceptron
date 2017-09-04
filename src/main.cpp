//Copyright (C) 2017 Aaron Mathankeri (aaron@quantifiedag.com)
//License: QuantifiedAG Software License.  See LICENSE.txt for full license.

/*
 *   \file main.cpp
 *   \brief Main source file
 *
 *  NOTE: - x0 & z0 are clamped at 1.0 for the biases!
 *          therefore for 3hidden nodes, there are 4
 */
#include <iostream>
#include "mkl.h"
#include "ioFunctions.hpp"
#include "parameters.hpp"
#include "initializations.hpp"
#include "feedForwardFunctions.hpp"
#include "backPropFunctions.hpp"
#include "gradientDescent.hpp"

using namespace std;

int main(int argc, char *argv[])
{
      cout << " Aaron's Back." << endl;

      //--------------------------------------------------------------------------------
      // declare variables for calculations
      const double learningRate = 1e-2;
      const int MAX_ITER = 1e5;

      double *X, *T, *Y; // inputs and targets
      double *x, *t, *y, *W, *V, *a, *z;

      double *outputErrors, *inputErrors;
      double *gradW, *gradV;

      X = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 ); // inputs
      T = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 ); // targets
      Y = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 ); // targets
      
      x = (double *)mkl_malloc( (DIMENSIONS + 1)*sizeof( double ), 64 ); // single pattern + bias
      t = (double *)mkl_malloc( NUM_OUTPUTS*sizeof( double ), 64 ); // single target
      y = (double *)mkl_malloc( NUM_OUTPUTS*sizeof( double ), 64 ); // outputs

      V = (double *)mkl_malloc( NUM_HIDDEN_NODES * (DIMENSIONS+1) *sizeof( double ), 64 ); //1st layer weights
      W = (double *)mkl_malloc( NUM_OUTPUTS * (NUM_HIDDEN_NODES+1)*sizeof( double ), 64 ); //2nd layer weights

      a = (double *)mkl_malloc( (NUM_HIDDEN_NODES)*sizeof( double ), 64 ); // activations 
      z = (double *)mkl_malloc( (NUM_HIDDEN_NODES + 1)*sizeof( double ), 64 ); // hidden nodes including bias

      outputErrors = (double *)mkl_malloc( NUM_OUTPUTS*sizeof( double ), 64 ); // 
      inputErrors = (double *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( double ), 64 ); // including bias

      gradV = (double *)mkl_malloc( NUM_HIDDEN_NODES*(DIMENSIONS+1)*sizeof( double ), 64 ); //1st layer weights
      gradW = (double *)mkl_malloc( NUM_OUTPUTS*(NUM_HIDDEN_NODES+1)*sizeof( double ), 64 ); //2nd layer weights

      memset( X, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( T, 0.0,  NUM_PATTERNS * sizeof(double));

      memset( x, 0.0,  (DIMENSIONS+1) * sizeof(double));
      memset( t, 0.0,  NUM_OUTPUTS * sizeof(double));
      memset( y, 0.0,  NUM_OUTPUTS * sizeof(double));

      memset( V, 0.0,  NUM_HIDDEN_NODES * (DIMENSIONS+1) *sizeof(double));
      memset( W, 0.0,  NUM_OUTPUTS * (NUM_HIDDEN_NODES+1)  * sizeof(double));

      memset( a, 0.0,  (NUM_HIDDEN_NODES) * sizeof(double));
      memset( z, 0.0,  (NUM_HIDDEN_NODES+1) * sizeof(double));

      memset( outputErrors, 0.0,  (NUM_OUTPUTS) * sizeof(double));
      memset( inputErrors, 0.0,  (NUM_HIDDEN_NODES+1) * sizeof(double));

      memset( gradV, 0.0,  NUM_HIDDEN_NODES *(DIMENSIONS+1) * sizeof(double));
      memset( gradW, 0.0,  NUM_OUTPUTS * (NUM_HIDDEN_NODES+1)  * sizeof(double));

      //--------------------------------------------------------------------------------
      //read data
      string inputsFile = "./data/xSquared/inputs.txt";
      string targetsFile = "./data/xSquared/targets.txt";

      loadData( X , inputsFile );
      loadData( T , targetsFile );

      cout << "Inputs" << endl;
      printVector( X, 10 );

      cout << "Targets" << endl;
      printVector( T, 10 );

      // Randomly initialize weights
      setRandomWeights( V,  NUM_HIDDEN_NODES, (DIMENSIONS+1) );
      setRandomWeights( W, NUM_OUTPUTS, (NUM_HIDDEN_NODES+1) );

      cout << "INITIAL WEIGHTS ARE.." << endl;
      cout << "\nFirst layer weights" << endl;
      printMatrix( V,  NUM_HIDDEN_NODES, (DIMENSIONS + 1) );

      cout << "\nSecond layer weights" << endl;
      printMatrix( W, NUM_OUTPUTS, (NUM_HIDDEN_NODES+1) );
      int iter = 0;

      cout << "\n\n TRAINING NETWORK USING SGD..." << endl;
      while (iter < MAX_ITER) {
	    iter++;
	    for (int i = 0; i < NUM_PATTERNS; ++i) {
		  memset( x, 0.0, (DIMENSIONS + 1)* sizeof(double));
		  augmentInput( X, i, x );
		  t[0] = T[i];
		  //--------------------------------------------------------------------------------
		  //FEEDFORWARD FUNCTIONS!!!!
		  memset( a, 0.0,  (NUM_HIDDEN_NODES) * sizeof(double));
		  memset( z, 0.0,  (NUM_HIDDEN_NODES+1) * sizeof(double));
		  memset( y, 0.0,  NUM_OUTPUTS * sizeof(double));
		  //cout << "\n\nFORWARD PROPAGATE THROUGH NETWORK " << endl;
		  //A. compute activations:
		  computeActivations( x, V, a);
		  //--------------------------------------------------------------------------------
		  //B. compute hidden nodes
		  computeHiddenUnits( a, z);
		  //--------------------------------------------------------------------------------
		  //C. compute output
		  computeOutputActivations( z , W, y );
		  //--------------------------------------------------------------------------------
		  // ERROR BACKPROPAGATION
		  //cout << "\n\nBACKPROPAGATE THROUGH NETWORK " << endl;
		  memset( outputErrors, 0.0,  (NUM_OUTPUTS) * sizeof(double));
		  memset( inputErrors, 0.0,  (NUM_HIDDEN_NODES+1) * sizeof(double));
		  memset( gradV, 0.0,  NUM_HIDDEN_NODES *(DIMENSIONS+1) * sizeof(double));
		  memset( gradW, 0.0,  NUM_OUTPUTS * (NUM_HIDDEN_NODES+1)  * sizeof(double));
		  //A. computeOutputErrors
		  computeOutputErrors( y, t, outputErrors );
		  //--------------------------------------------------------------------------------
		  //B. computeInput errors
		  computeHiddenErrors( a, W, outputErrors, inputErrors );
		  //--------------------------------------------------------------------------------
		  //C. find gradient
		  computeGradW( outputErrors, z , gradW );
		  computeGradV( inputErrors, x, gradV );
		  //--------------------------------------------------------------------------------
		  // STOCHASTIC GRADIENT DESCENT
		  //A. apply gradient
		  //multiply weights by learningrate
		  applyLearningRate( gradW, (NUM_OUTPUTS*(NUM_HIDDEN_NODES+1)), learningRate );
		  applyLearningRate( gradV, (NUM_HIDDEN_NODES*(DIMENSIONS+1)), learningRate );
		  //--------------------------------------------------------------------------------
		  //B. update W = W - eta*gradW
		  updateWeights( W, gradW, (NUM_OUTPUTS*(NUM_HIDDEN_NODES+1)) );
		  updateWeights( V, gradV, (NUM_HIDDEN_NODES*(DIMENSIONS+1)));
		  //--------------------------------------------------------------------------------
	    }
      }
      cout << "complete." << endl;
      
      //--------------------------------------------------------------------------------
      for (int i = 0; i < NUM_PATTERNS; ++i) {
	    memset( x, 0.0, (DIMENSIONS + 1)* sizeof(double));
	    augmentInput( X, i, x );
	    t[0] = T[i];
	    //--------------------------------------------------------------------------------
	    //FEEDFORWARD FUNCTIONS!!!!
	    memset( a, 0.0,  (NUM_HIDDEN_NODES) * sizeof(double));
	    memset( z, 0.0,  (NUM_HIDDEN_NODES+1) * sizeof(double));
	    memset( y, 0.0,  NUM_OUTPUTS * sizeof(double));
	    //cout << "\n\nFORWARD PROPAGATE THROUGH NETWORK " << endl;
	    //A. compute activations:
	    computeActivations( x, V, a);
	    //--------------------------------------------------------------------------------
	    //B. compute hidden nodes
	    computeHiddenUnits( a, z);
	    //--------------------------------------------------------------------------------
	    //C. compute output
	    computeOutputActivations( z , W, y );
	    cout << "For Input : "<< x[1]<< "\t Target is " << t[0] << "\t Output is  : " << y[0] << endl;
	    //--------------------------------------------------------------------------------
	    Y[i] = y[0];
      }
      cout << "least squares error is " << computeLeastSquaresError( T , Y ) << endl;
      //--------------------------------------------------------------------------------
      printf ("\n Deallocating memory \n\n");
      mkl_free( X );
      mkl_free( T );
      mkl_free( Y );
      mkl_free( x );
      mkl_free( t );
      mkl_free( y );
      mkl_free( W );
      mkl_free( V );
      mkl_free( a );
      mkl_free( z );
      mkl_free( outputErrors );
      mkl_free( inputErrors );
      mkl_free( gradW );
      mkl_free( gradV );
      printf (" Example completed. \n\n");

      return 0;
}
