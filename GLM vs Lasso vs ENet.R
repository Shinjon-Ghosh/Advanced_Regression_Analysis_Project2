
library(glmnet)
library(tidyverse)

`%nin%` <- Negate(`%in%`)

# True variable indices
true_vars_list <- list(
  y1 = c(1, 20, 40, 60, 80, 90),
  y2 = c(1, 20, 40, 60, 80, 90),
  y3 = c(1, 20, 40, 60, 80, 90)
)

# Your previously generated data
data <- full_data

results_all <- data.frame()

for (i in 0:29) {
  for (target in c("y1", "y2", "y3")) {
    start_idx <- i * 200 + 1
    end_idx <- start_idx + 199
    
    y <- data[[target]][start_idx:end_idx]
    X_raw <- data[start_idx:end_idx, 4:103]
    
    # ---- Clean X ----
    X <- as.matrix(X_raw)
    
    # Remove columns with NA, NaN, Inf, or zero variance
    valid_cols <- apply(X, 2, function(col) {
      all(is.finite(col)) && sd(col) > 0
    })
    X <- X[, valid_cols]
    
    true_vars <- true_vars_list[[target]]
    
    # ---- POISSON GLM ----
    tryCatch({
      mm <- model.matrix(~ ., data = as.data.frame(X))
      mm <- mm[, apply(mm, 2, function(col) all(is.finite(col)) && sd(col) > 0)]
      fit_glm <- glm(y ~ mm - 1, family = poisson)  # remove intercept since included in mm
      selected_glm <- which(coef(fit_glm) != 0)
      selected_glm_names <- names(coef(fit_glm))[selected_glm]
      selected_glm_idx <- as.integer(gsub("x", "", gsub("mm", "", selected_glm_names)))
      selected_glm_idx <- selected_glm_idx[!is.na(selected_glm_idx)]
    }, error = function(e) {
      selected_glm_idx <- integer(0)  # in case glm fails
    })
    
    # ---- LASSO ----
    fit_lasso <- cv.glmnet(X, y, alpha = 1, family = "poisson")
    selected_lasso <- which(coef(fit_lasso, s = "lambda.min")[-1] != 0)
    
    # ---- ELASTIC NET ----
    fit_enet <- cv.glmnet(X, y, alpha = 0.5, family = "poisson")
    selected_enet <- which(coef(fit_enet, s = "lambda.min")[-1] != 0)
    
    # --- Evaluation helper
    eval_selection <- function(selected, true_vars) {
      TP <- sum(true_vars %in% selected)
      FP <- sum(!(1:100 %in% true_vars) & (1:100 %in% selected))
      FN <- sum(true_vars %nin% selected)
      TN <- sum(!(1:100 %in% true_vars) & !(1:100 %in% selected))
      FPR <- if ((FP + TN) == 0) NA else FP / (FP + TN)
      FNR <- if ((FN + TP) == 0) NA else FN / (FN + TP)
      return(data.frame(TP, FP, FN, TN, FPR, FNR))
    }
    
    results_all <- rbind(
      results_all,
      cbind(eval_selection(selected_glm_idx, true_vars), response = target, dataset = i+1, model = "GLM"),
      cbind(eval_selection(selected_lasso, true_vars), response = target, dataset = i+1, model = "LASSO"),
      cbind(eval_selection(selected_enet, true_vars), response = target, dataset = i+1, model = "ElasticNet")
    )
  }
}

# -------- Summary Table --------
summary_table <- results_all %>%
  group_by(response, model) %>%
  summarise(across(c(TP, FP, FN, TN, FPR, FNR), mean), .groups = "drop")

print(summary_table)

# -------- Boxplots --------
ggplot(results_all, aes(x = model, y = FNR, fill = model)) +
  geom_boxplot() +
  facet_wrap(~response) +
  labs(title = "False Negative Rate by Model and Response", y = "FNR") +
  theme_minimal()

ggplot(results_all, aes(x = model, y = FPR, fill = model)) +
  geom_boxplot() +
  facet_wrap(~response) +
  labs(title = "False Positive Rate by Model and Response", y = "FPR") +
  theme_minimal()
