# =============================================================================
# Clear workspace
rm(list=ls())

set.seed(123)
# =============================================================================
setwd('/Users/aaron/playground/proj-MultilayerPerceptron/data/')
# =============================================================================
# libraries
library(ggplot2)
# =============================================================================
# sample from uniform distribution
# x ~ U(0,1)
nSamples <- 50
inputs <- runif( nSamples, -1, 1 )
# =============================================================================
# Targets
# t = sin(2*pi*x) + noise
targets <- inputs^2
# =============================================================================
# create dataframe for plots
df <- cbind.data.frame( inputs, targets )
# =============================================================================
# define functions to plot with data
#f1 <- function(x) {0.929 + -1.767*x} # found from running c++ file
#f3 <- function(x) {-0.031 + 11.717*x + -35.138*x^2 + 23.758*x^3 } # found from running c++ file
# =============================================================================
p <- ggplot(df, aes(x = inputs, y = targets)) +
  geom_point()

#p + stat_function(fun = f1, aes( colour = "M=1")) +
#  stat_function(fun = f3, aes(colour = "M=3")) +
#  scale_colour_manual("Functions",values = c("red", "blue"), 
#                      labels = c("M=1", "M=3"))
# =============================================================================
# write to file
write(inputs, file = "xSquared/inputs.txt", ncolumns = 1)
write(targets, file = "xSquared/targets.txt", ncolumns = 1)
