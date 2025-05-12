setwd("F:/Applied Statistics Course - ISU/Advanced Regression Analysis/Project/Project 2")

data <-read.csv(file = "Data/simulated_poisson_6000rows.csv")

`%nin%` <- Negate(`%in%`)

# Define the true variable indices for each target
true_vars_list <- list(
  y1 = c(1, 20, 40, 60, 80, 90),
  y2 = c(1, 20, 40, 60, 80, 90),
  y3 = c(1, 20, 40, 60, 80, 90)
)
# Load required libraries
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Stan model code for Normal prior
normal_model_code <- "
data {
  int<lower=1> n;
  int<lower=1> p;
  matrix[n, p] X;
  int<lower=0> y[n];
}
parameters {
  real alpha;
  vector[p] beta;
}
model {
  beta ~ normal(0, 1);
  alpha ~ normal(0, 5);
  y ~ poisson_log(X * beta + alpha);
}
"

# Stan model code for Laplace prior
laplace_model_code <- "
functions {
  real laplace_lpdf(real x, real mu, real b) {
    return log(0.5 / b) - fabs(x - mu) / b;
  }
}
data {
  int<lower=1> n;
  int<lower=1> p;
  matrix[n, p] X;
  int<lower=0> y[n];
}
parameters {
  real alpha;
  vector[p] beta;
}
model {
  for (j in 1:p)
    beta[j] ~ laplace(0, 1);
  alpha ~ normal(0, 5);
  y ~ poisson_log(X * beta + alpha);
}
"

# Compile Stan models once
model_normal <- stan_model(model_code = normal_model_code)
model_laplace <- stan_model(model_code = laplace_model_code)

# Initialize result containers
results_bayes <- data.frame()
selection_record <- list()

for (i in 0:29) {
  for (target in c("y1", "y2", "y3")) {
    
    start_idx <- i * 200 + 1
    end_idx <- start_idx + 99
    y <- data[[target]][start_idx:end_idx]
    X <- as.matrix(data[start_idx:end_idx, 4:103])
    stan_data <- list(n = nrow(X), p = ncol(X), X = X, y = y)
    true_vars <- true_vars_list[[target]]
    
    dataset_label <- paste0(target, "_dataset", i + 1)
    
    # ------------------------
    # Normal Prior
    fit_normal <- sampling(model_normal, data = stan_data, iter = 3000, chains = 2, seed = 123)
    beta_draws_normal <- extract(fit_normal)$beta
    ci_normal <- apply(beta_draws_normal, 2, quantile, probs = c(0.025, 0.975))
    selected_normal <- which(ci_normal[1, ] * ci_normal[2, ] > 0)
    selected_vector_normal <- rep(0, 100)
    selected_vector_normal[selected_normal] <- 1
    
    # Evaluation
    TP <- sum(true_vars %in% selected_normal)
    FP <- sum(!(1:100 %in% true_vars) & (1:100 %in% selected_normal))
    FN <- sum(true_vars %in% setdiff(1:100, selected_normal))
    TN <- sum(!(1:100 %in% true_vars) & !(1:100 %in% selected_normal))
    TPR <- if ((TP + FN) == 0) NA else TP / (TP + FN)
    TNR <- if ((TN + FP) == 0) NA else TN / (TN + FP)
    
    results_bayes <- rbind(results_bayes, data.frame(
      response = target, dataset = i + 1, model = "NormalPrior",
      TP = TP, FP = FP, FN = FN, TN = TN, TPR = TPR, TNR = TNR
    ))
    
    selection_record[[paste0(dataset_label, "_Normal")]] <- selected_vector_normal
    
    # ------------------------
    # Laplace Prior
    fit_laplace <- sampling(model_laplace, data = stan_data, iter = 3000, chains = 2, seed = 123)
    beta_draws_laplace <- extract(fit_laplace)$beta
    ci_laplace <- apply(beta_draws_laplace, 2, quantile, probs = c(0.025, 0.975))
    selected_laplace <- which(ci_laplace[1, ] * ci_laplace[2, ] > 0)
    selected_vector_laplace <- rep(0, 100)
    selected_vector_laplace[selected_laplace] <- 1
    
    TP <- sum(true_vars %in% selected_laplace)
    FP <- sum(!(1:100 %in% true_vars) & (1:100 %in% selected_laplace))
    FN <- sum(true_vars %in% setdiff(1:100, selected_laplace))
    TN <- sum(!(1:100 %in% true_vars) & !(1:100 %in% selected_laplace))
    TPR <- if ((TP + FN) == 0) NA else TP / (TP + FN)
    TNR <- if ((TN + FP) == 0) NA else TN / (TN + FP)
    
    results_bayes <- rbind(results_bayes, data.frame(
      response = target, dataset = i + 1, model = "LaplacePrior",
      TP = TP, FP = FP, FN = FN, TN = TN, TPR = TPR, TNR = TNR
    ))
    
    selection_record[[paste0(dataset_label, "_Laplace")]] <- selected_vector_laplace
    
    cat(target, "_", i + 1, ": Done\n")
  }
}

# Save both objects to RDS files
saveRDS(results_bayes, file = "results_bayes.rds")
saveRDS(selection_record, file = "selection_record_bayes.rds")



library(dplyr)
library(ggplot2)
library(reshape2)

# Combine selection_record list to matrix
selection_df <- do.call(rbind, selection_record)

# Parse labels into response and model
meta_info <- do.call(rbind, strsplit(rownames(selection_df), "_"))
selection_df <- as.data.frame(selection_df)
selection_df$response <- meta_info[, 1]
selection_df$dataset <- meta_info[, 2]
selection_df$model <- meta_info[, 3]

# Gather to long format
selection_long <- melt(selection_df,
                       id.vars = c("response", "dataset", "model"),
                       variable.name = "variable",
                       value.name = "selected")

# Convert variable to numeric (e.g., x1 → 1)
selection_long$variable <- as.integer(sub("V", "", selection_long$variable))

# Count frequency for each (response, model, variable)
freq_summary <- selection_long %>%
  group_by(response, model, variable) %>%
  summarise(Frequency = sum(selected), .groups = "drop")

# For better ordering of y-axis
freq_summary$variable <- factor(
  paste0("x", freq_summary$variable),
  levels = paste0("x", rev(1:100))  # reversed for y-axis top-to-bottom
)
freq_summary$model <- factor(freq_summary$model, levels = c("Normal", "Laplace"))


pdf(file ="Figure/VB_Freq_bayes.pdf")
op<-par(mar=c(5,5,4,1),cex.axis=1.5,font.axis=2,font.lab=2,cex.main=2,cex.lab=1.6)
ggplot(freq_summary, aes(x = model, y = variable, fill = Frequency)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  facet_wrap(~ response) +
  labs(
    #title = "Variable Selection Frequency Across 30 Datasets",
    x = "Model",
    y = "Variable",
    fill = "Frequency"
  ) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 6))
par(op)
dev.off()

library(dplyr)

results_bayes_summary <- results_bayes %>%
  group_by(response, model) %>%
  summarise(
    Mean_TP = mean(TP),
    Mean_FP = mean(FP),
    Mean_FN = mean(FN),
    Mean_TN = mean(TN),
    Mean_TPR = mean(TPR, na.rm = TRUE),
    Mean_TNR = mean(TNR, na.rm = TRUE),
    .groups = "drop"
  )

print(results_bayes_summary)

library(ggplot2)


results_bayes$model <- factor(results_bayes$model, levels = c("NormalPrior", "LaplacePrior"))
pdf(file ="Figure/box_TPR_bayes.pdf")
op<-par(mar=c(5,5,4,1),cex.axis=1.5,font.axis=2,font.lab=2,cex.main=2,cex.lab=1.6)
ggplot(results_bayes, aes(x = model, y = TPR, fill = model)) +
  geom_boxplot() +
  facet_wrap(~ response) +
  #labs(title = "True Positive Rate (TPR) by Response and Model", y = "TPR", x = "") +
  theme_minimal() +
  theme(legend.position = "none")
par(op)
dev.off()

# TNR Boxplot
pdf(file ="Figure/box_TNR_bayes.pdf")
op<-par(mar=c(5,5,4,1),cex.axis=1.5,font.axis=2,font.lab=2,cex.main=2,cex.lab=1.6)
ggplot(results_bayes, aes(x = model, y = TNR, fill = model)) +
  geom_boxplot() +
  facet_wrap(~ response) +
  #labs(title = "True Negative Rate (TNR) by Response and Model", y = "TNR", x = "") +
  theme_minimal() +
  theme(legend.position = "none")
par(op)
dev.off()


# From your Stan fits
beta_normal <- extract(fit_normal)$beta  # matrix: draws × 100
beta_laplace <- extract(fit_laplace)$beta

# Posterior summaries
summary_normal <- apply(beta_normal, 2, function(x) c(mean = mean(x), sd = sd(x),
                                                      lower = quantile(x, 0.025),
                                                      upper = quantile(x, 0.975)))

summary_laplace <- apply(beta_laplace, 2, function(x) c(mean = mean(x), sd = sd(x),
                                                        lower = quantile(x, 0.025),
                                                        upper = quantile(x, 0.975)))

# Combine into data frame
coef_df <- data.frame(
  variable = paste0("x", 1:100),
  mean_normal = summary_normal["mean", ],
  mean_laplace = summary_laplace["mean", ],
  lower_normal = summary_normal["lower.2.5%", ],
  upper_normal = summary_normal["upper.97.5%", ],
  lower_laplace = summary_laplace["lower.2.5%", ],
  upper_laplace = summary_laplace["upper.97.5%", ]
)

library(ggplot2)
ggplot(coef_df, aes(x = mean_normal, y = mean_laplace, label = variable)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "Posterior Mean Comparison",
       x = "Normal Prior", y = "Laplace Prior") +
  theme_minimal()

library(tidyr)

coef_long <- coef_df %>%
  select(variable, mean_normal, mean_laplace) %>%
  pivot_longer(cols = -variable, names_to = "model", values_to = "posterior_mean")

ggplot(coef_long, aes(x = variable, y = posterior_mean, color = model, group = model)) +
  geom_line() +
  labs(title = "Posterior Coefficients by Variable",
       x = "Variable", y = "Posterior Mean") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90))


# Identify selected variables (CI does not include 0)
selected_normal <- with(coef_df, which(lower_normal * upper_normal > 0))
selected_laplace <- with(coef_df, which(lower_laplace * upper_laplace > 0))

intersect_vars <- intersect(selected_normal, selected_laplace)
unique_normal <- setdiff(selected_normal, selected_laplace)
unique_laplace <- setdiff(selected_laplace, selected_normal)

cat("Common selected variables:", paste0("x", intersect_vars), "\n")
cat("Only in normal:", paste0("x", unique_normal), "\n")
cat("Only in laplace:", paste0("x", unique_laplace), "\n")


