/**
 *   \file initializations.hpp
 *   \brief A Documented file.
 *
 *  Helper functions needed before 
 *  network inference
 */

#ifndef INITIALIZATIONS_H
#define INITIALIZATIONS_H

#include <iostream>
#include "parameters.hpp"
#include "mkl.h"

using namespace std;

//-----------------------------------------------------
//initialize to avoid memory errors
void computeDataMatrix( const double *x, double *X );

void initializeMatrix( double * Matrix, int rows, int columns );
double fRand(double fMin, double fMax);
void setRandomWeights( double * weights, int nRows, int nCols );

#endif /* INITIALIZATIONS_H */
