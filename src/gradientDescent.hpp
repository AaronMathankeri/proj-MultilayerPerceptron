/**
 *   \file gradientDescent.hpp
 *   \brief Gradient descent
 *
 *  Inference procedure for calculating
 * gradient of a network topology
 */

#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include "parameters.hpp"
#include "mkl.h"
#include <cstring>
void applyLearningRate( double *x, const int nElements, const double learningRate );
void updateWeights( double *weights, const double *deltaWeights, const int nElements );

#endif /* GRADIENTDESCENT_H */
