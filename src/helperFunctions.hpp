/**
 *   \file helperFunctions.hpp
 *   \brief Functions independent of network
 *
 *  Defn of activation functions should go here
 * easy to use different defns without breaking
 * network
 */

#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H

#include "parameters.hpp"
#include <math.h>

double logisticSigmoid( const double a );
double dSigmoid( const double a );

#endif /* HELPERFUNCTIONS_H */
