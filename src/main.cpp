//Copyright (C) 2017 Aaron Mathankeri (aaron@quantifiedag.com)
//License: QuantifiedAG Software License.  See LICENSE.txt for full license.

/*
 *   main.cpp
 *   \brief HPC implementation of simple neural network model
 *
 *  p(c=1|x) = sigma( x*w )
 * 
 */
#include <iostream>
#include "mkl.h"
#include "ioFunctions.hpp"
#include "parameters.hpp"

using namespace std;

int main(int argc, char *argv[])
{
      cout << " Aaron's Back." << endl;

      //--------------------------------------------------------------------------------
      // declare variables for calculations
      double *x1, *x2, *t, *X; //data
      double *weights, *y, *designMatrix, *R; //logistic regression parameters

      x1 = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );
      x2 = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );
      t = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );
      X = (double *)mkl_malloc( NUM_PATTERNS*(ORDER - 1)*sizeof( double ), 64 );
      
      weights = (double *)mkl_malloc( ORDER*sizeof( double ), 64 );
      y = (double *)mkl_malloc( NUM_PATTERNS*sizeof( double ), 64 );
      designMatrix = (double *)mkl_malloc( NUM_PATTERNS*ORDER*sizeof( double ), 64 );
      R = (double *)mkl_malloc( NUM_PATTERNS*NUM_PATTERNS*sizeof( double ), 64 );
      
      memset( x1, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( x2, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( t, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( X, 0.0,  NUM_PATTERNS *(ORDER - 1)* sizeof(double));

      memset( weights, 0.0,  ORDER * sizeof(double));
      memset( y, 0.0,  NUM_PATTERNS * sizeof(double));
      memset( designMatrix, 0.0,  NUM_PATTERNS * ORDER* sizeof(double));
      memset( R, 0.0,  NUM_PATTERNS *NUM_PATTERNS* sizeof(double));
      //--------------------------------------------------------------------------------
      //read data
      string x1File = "./data/train/x1.txt";
      string x2File = "./data/train/x2.txt";
      string targetsFile = "./data/train/class.txt";

      loadData( x1 , x1File );
      loadData( x2 , x2File );
      loadData( t , targetsFile );
      
      cout << "First 10 Features" << endl;
      cout << "x1\t | \tx2" << endl;
      printFeatures( x1, x2, 5 );
      
      //cout << "\nFirst 10 Targets" << endl;
      //printVector( t, 10 );
      //--------------------------------------------------------------------------------
      //1. Randomly initialize weights.( srandnull() ) for true randomization
      //setRandomWeights( weights );
      //cout << "\nInitial Weights" << endl;
      //printVector( weights, ORDER );
      //--------------------------------------------------------------------------------


      printf ("\n Deallocating memory \n\n");
      mkl_free( x1 );
      mkl_free( x2 );
      mkl_free( t );
      mkl_free( weights );
      mkl_free( y );
      mkl_free( designMatrix );
      mkl_free( R );
      mkl_free( X );
      printf (" Example completed. \n\n");

      return 0;
}
