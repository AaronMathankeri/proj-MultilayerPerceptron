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



#endif /* BACKPROPFUNCTIONS_H */
