/**
 *   \file backPropFunctions.hpp
 *   \brief Erro Backpropagation 
 *
 *  Functions used to calculate network gradient
 *
 */

#ifndef BACKPROPFUNCTIONS_H
#define BACKPROPFUNCTIONS_H

#include "mkl.h"
#include "parameters.hpp"
#include "helperFunctions.hpp"

void computeOutputErrors( const double *y, const double *t, double *outputErrors);
void computeHiddenErrors( const double *a, const double *V, const double *outputErrors, double *inputErrors );

void computeGradV( const double *outputErrors, const double *z, double *gradV );
void computeGradW( const double *inputErrors, const double *x, double *gradW );
#endif /* BACKPROPFUNCTIONS_H */
