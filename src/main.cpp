//Copyright (C) 2017 Aaron Mathankeri (aaron@quantifiedag.com)
//License: QuantifiedAG Software License.  See LICENSE.txt for full license.

/*
 *   \file example.cpp
 *   \brief A Documented file.
 *
 *  NOTE: x0 & z0 are clamped at 1.0 for the biases!
 *        therefore for 3hidden nodes, there are 4
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
      const double learningRate = 1e-4;
      const int MAX_ITER = 50;

      double *X, *T; // inputs and targets
      double *x, *t, *y, *W, *V, *a, *z;

      double *outputErrors, *inputErrors;
      double *gradW, *gradV;

      X = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 ); // inputs
      T = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 ); // targets

      x = (double *)mkl_malloc( (DIMENSIONS + 1)*sizeof( double ), 64 ); // single pattern + bias
      t = (double *)mkl_malloc( NUM_OUTPUTS*sizeof( double ), 64 ); // single target
      y = (double *)mkl_malloc( NUM_OUTPUTS*sizeof( double ), 64 ); // outputs

      W = (double *)mkl_malloc( NUM_HIDDEN_NODES * (DIMENSIONS+1) *sizeof( double ), 64 ); //1st layer weights
      V = (double *)mkl_malloc( NUM_OUTPUTS * (NUM_HIDDEN_NODES+1)*sizeof( double ), 64 ); //2nd layer weights

      a = (double *)mkl_malloc( (NUM_HIDDEN_NODES)*sizeof( double ), 64 ); // activations 
      z = (double *)mkl_malloc( (NUM_HIDDEN_NODES + 1)*sizeof( double ), 64 ); // hidden nodes including bias
      
      outputErrors = (double *)mkl_malloc( NUM_OUTPUTS*sizeof( double ), 64 ); // 
      inputErrors = (double *)mkl_malloc( (NUM_HIDDEN_NODES+1)*sizeof( double ), 64 ); // including bias

      gradW = (double *)mkl_malloc( (DIMENSIONS + 1)*NUM_HIDDEN_NODES*sizeof( double ), 64 ); //1st layer weights
      gradV = (double *)mkl_malloc( NUM_OUTPUTS*(NUM_HIDDEN_NODES+1)*sizeof( double ), 64 ); //2nd layer weights
      
      memset( X, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( T, 0.0,  NUM_PATTERNS * sizeof(double));

      memset( x, 0.0,  (DIMENSIONS+1) * sizeof(double));
      memset( t, 0.0,  NUM_OUTPUTS * sizeof(double));
      memset( y, 0.0,  NUM_OUTPUTS * sizeof(double));

      memset( W, 0.0,  NUM_HIDDEN_NODES * (DIMENSIONS+1) *sizeof(double));
      memset( V, 0.0,  NUM_OUTPUTS * (NUM_HIDDEN_NODES+1)  * sizeof(double));

      memset( a, 0.0,  (NUM_HIDDEN_NODES) * sizeof(double));
      memset( z, 0.0,  (NUM_HIDDEN_NODES+1) * sizeof(double));

      memset( gradW, 0.0,  (DIMENSIONS + 1) * NUM_HIDDEN_NODES* sizeof(double));
      memset( gradV, 0.0,  NUM_OUTPUTS * NUM_HIDDEN_NODES  * sizeof(double));
      //--------------------------------------------------------------------------------
      //read data
      string inputsFile = "./data/xSquared/inputs.txt";
      string targetsFile = "./data/xSquared/targets.txt";

      loadData( X , inputsFile );
      loadData( T , targetsFile );
      //*
      cout << "Inputs" << endl;
      printVector( X, 10 );

      cout << "Targets" << endl;
      printVector( T, 10 );
      //*/

      //3. Randomly initialize weights
      setRandomWeights( W,  NUM_HIDDEN_NODES, (DIMENSIONS+1) );
      setRandomWeights( V, NUM_OUTPUTS, (NUM_HIDDEN_NODES+1) );
      cout << "INITIAL WEIGHTS ARE.." << endl;
      cout << "\nFirst layer weights" << endl;
      printMatrix( W,  NUM_HIDDEN_NODES, (DIMENSIONS + 1) );

      cout << "\nSecond layer weights" << endl;
      printMatrix( V, NUM_OUTPUTS, (NUM_HIDDEN_NODES+1) );

      //--------------------------------------------------------------------------------
      //2. each input pattern must have a x0 clamped at 1.0 for the bias
      // same for the hidden layer

      for (int i = 0; i < 1; ++i) {
	    augmentInput( X, i, x );
	    t[0] = T[i];
	    //cout << "x is " << x[0] << "\t" << x[1] << endl;
	    cout << "\nx is " << "\n";
	    printVector( x , DIMENSIONS+1);
	    cout << "t is " << "\n";
	    printVector( t , NUM_OUTPUTS);

      }

      //--------------------------------------------------------------------------------
      //FEEDFORWARD FUNCTIONS!!!!
      cout << "\n\nFEEDFORWARD CALCULATIONS!!!.." << endl;
      memset( a, 0.0,  (NUM_HIDDEN_NODES) * sizeof(double));
      memset( z, 0.0,  (NUM_HIDDEN_NODES+1) * sizeof(double));
      memset( y, 0.0,  NUM_OUTPUTS * sizeof(double));
      //cout << "\n\nFORWARD PROPAGATE THROUGH NETWORK " << endl;
      //A. compute activations:
      cout << "\nComputing activations" << endl;
      computeActivations( x, W, a);
      cout << "Activations : " << endl;
      printVector( a, NUM_HIDDEN_NODES );

      //--------------------------------------------------------------------------------
      //B. compute hidden nodes
      cout << "\nComputing hidden nodes" << endl;
      computeHiddenUnits( a, z);
      cout << "Hidden Nodes : " << endl;
      printVector( z, NUM_HIDDEN_NODES+1 );

      //--------------------------------------------------------------------------------
      //C. compute output
      cout << "\nComputing outputs" << endl;
      computeOutputActivations( z , V, y );
      cout << "Output is  : " << endl;
      printVector( y, NUM_OUTPUTS );
      //--------------------------------------------------------------------------------
      // ERROR BACKPROPAGATION
      memset( outputErrors, 0.0,  NUM_OUTPUTS * sizeof(double));
      memset( inputErrors, 0.0,  (NUM_HIDDEN_NODES+1) * sizeof(double));

      cout << "\n\nBACKPROPAGATE THROUGH NETWORK " << endl;
      //A. computeOutputErrors
      cout << "\nComputing output errors" << endl;
      computeOutputErrors( y, t, outputErrors );
      cout << "Output Errors are  : " << endl;
      printVector( outputErrors, NUM_OUTPUTS );
      //--------------------------------------------------------------------------------
      //B. computeInput errors
      cout << "\nComputing input errors" << endl;
      computeHiddenErrors( a, V, outputErrors, inputErrors );
      cout << "Input Errors are  : " << endl;
      printVector( inputErrors, NUM_HIDDEN_NODES+1 );
      //--------------------------------------------------------------------------------
      /*
      computeGradV( outputErrors, z , gradV );
      computeGradW( inputErrors, x, gradW );
      cout << "complete." << endl;
      /*
	cout << "\n\n GradV   : " << endl;
	printMatrix( gradV, NUM_OUTPUTS ,NUM_HIDDEN_NODES );
	cout << "\n\n GradW  : " << endl;
	printMatrix( gradW, NUM_HIDDEN_NODES, (DIMENSIONS+1) );
      */
      //--------------------------------------------------------------------------------
      // STOCHASTIC GRADIENT DESCENT
      //1. get gradient
      //2. multiply by learning rate
      //3. subtract current weights
      //cout << "\n\nSTOCHASTIC GRADIENT DESCENT " << endl;
      /*
      //--------------------------------------------------------------------------------
      cout << "Updating weights using SGD" << endl;
      //multiply weights by learningrate
      applyLearningRate( gradV, (NUM_OUTPUTS*NUM_HIDDEN_NODES), learningRate );
      applyLearningRate( gradW, (NUM_HIDDEN_NODES*(DIMENSIONS+1)), learningRate );
      /*
	cout << "\n\n GradV   : " << endl;
	printMatrix( gradV, NUM_OUTPUTS ,NUM_HIDDEN_NODES );
	cout << "\n\n GradW  : " << endl;
	printMatrix( gradW, NUM_HIDDEN_NODES, (DIMENSIONS+1) );
      */
      /*
      //--------------------------------------------------------------------------------
      updateWeights( W, gradW, (NUM_HIDDEN_NODES*(DIMENSIONS+1)));
      updateWeights( V, gradV, (NUM_OUTPUTS*NUM_HIDDEN_NODES) );
      cout << "complete." << endl;
      /*
	cout << "NEW Weights are.." << endl;
	cout << "\nFirst layer weights" << endl;
	printMatrix( W,  NUM_HIDDEN_NODES, (DIMENSIONS + 1) );

	cout << "\nSecond layer weights" << endl;
	printMatrix( V, NUM_OUTPUTS, NUM_HIDDEN_NODES );
      */
      //--------------------------------------------------------------------------------


      /*
      cout << "FINAL WEIGHTS ARE.." << endl;
      cout << "\nFirst layer weights" << endl;
      printMatrix( W,  NUM_HIDDEN_NODES, (DIMENSIONS + 1) );

      cout << "\nSecond layer weights" << endl;
      printMatrix( V, NUM_OUTPUTS, NUM_HIDDEN_NODES );

      for (int i = 0; i < NUM_PATTERNS; ++i) {
	    memset( xPrime, 0.0, (DIMENSIONS + 1)* sizeof(double));
	    augmentInput( x, i, xPrime );
	    //--------------------------------------------------------------------------------
	    //FEEDFORWARD FUNCTIONS!!!!
	    computeActivations( xPrime, W, a);
	    //--------------------------------------------------------------------------------
	    computeHiddenUnits( a, z);
	    //--------------------------------------------------------------------------------
	    computeOutputActivations( z , V, y );
	    cout << "\n\nFor Input : "<< xPrime[1]<< "\t Target is " << t[i] << "\t Output is  : " << y[0] << endl;
	    //printVector( y, NUM_OUTPUTS );
      }
      //*/
      //--------------------------------------------------------------------------------
      printf ("\n Deallocating memory \n\n");
      mkl_free( X );
      mkl_free( T );



      mkl_free( y );
      mkl_free( W );
      mkl_free( V );
      mkl_free( a );
      mkl_free( z );
      //mkl_free( outputErrors );
      // mkl_free( inputErrors );
      // mkl_free( gradW );
      //mkl_free( gradV );
      printf (" Example completed. \n\n");

      return 0;
}
