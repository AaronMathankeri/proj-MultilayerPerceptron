/**
 *   \file feedForwardFunctions.hpp
 *   \brief A Documented file.
 *
 *  All functions needed to evaluate
 *  forward pass of network
 */

#ifndef FEEDFORWARDFUNCTIONS_H
#define FEEDFORWARDFUNCTIONS_H

#include "mkl.h"
#include "parameters.hpp"
#include "helperFunctions.hpp"
//---------------------------------------------------------------------------------
//feedforward functions:
void computeActivations( const double *x, const double *V, double *a);
void computeHiddenUnits( const double *a, double* z);
void computeOutputActivations( const double *z, const double *W, double *y);

#endif /* FEEDFORWARDFUNCTIONS_H */
