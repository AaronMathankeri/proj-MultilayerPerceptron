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
void computeHiddenErrors( const double *a, const double *W, const double *outputErrors, double *inputErrors );

void computeGradW( const double *outputErrors, const double *z, double *gradW );
void computeGradV( const double *inputErrors, const double *x, double *gradV );
#endif /* BACKPROPFUNCTIONS_H */
