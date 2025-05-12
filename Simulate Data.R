setwd("F:/Applied Statistics Course - ISU/Advanced Regression Analysis/Project/Project 2")

library(MASS)
library(Matrix)
library(tidyverse)

set.seed(123)

n <- 200        # observations per dataset
p <- 100        # number of predictors
num_datasets <- 30
true_vars <- c(1, 20, 40, 60, 80, 90)

# Initialize container for all datasets
full_data <- data.frame()

for (i in 1:num_datasets) {
  # Simulate design matrix X
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  colnames(X) <- paste0("x", 1:p)
  
  # --- y1: Sparse linear ---
  mu1 <- 0.25 + 0.5*X[,1] + 0.1*X[,20] + 0.2*X[,40] + 0.7*X[,60] + 1.2*X[,80] + 1.4*X[,90]
  y1 <- rpois(n, lambda = exp(mu1))
  
  # --- y2: Add interaction ---
  mu2 <- mu1 + X[,1]*X[,40] - 1.7 * X[,60]*X[,90]
  y2 <- rpois(n, lambda = exp(mu2))
  
  # --- y3: Add nonlinearity ---
  mu3 <- mu2 + 0.2 * X[,40]^2
  y3 <- rpois(n, lambda = exp(mu3))
  
  # Combine into a dataset: y1, y2, y3, x1 to x100
  dataset <- data.frame(y1 = y1, y2 = y2, y3 = y3, X)
  
  full_data <- rbind(full_data, dataset)
}

# Save the full dataset
write.csv(full_data, file = "Data/simulated_poisson_6000rows.csv", row.names = FALSE)
saveRDS(full_data, file = "Data/simulated_poisson_6000rows.rds")
