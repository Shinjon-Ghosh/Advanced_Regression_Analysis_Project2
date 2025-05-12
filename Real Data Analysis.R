# Load required packages
library(readxl)
library(glmnet)
library(dplyr)

# Load data
data <- read_excel("F:\\Applied Statistics Course - ISU\\Advanced Regression Analysis\\Project\\Project 2\\day.xlsx")

# Prepare data
X <- data %>%
  select(-instant, -dteday, -casual, -registered, -cnt, -yr, -atemp) %>%
  as.data.frame()

y <- data$cnt

# Convert to matrix for glmnet
X_matrix <- model.matrix(~ ., X)[, -1]  # remove intercept added by model.matrix
y_vector <- as.numeric(y)

# Lasso Regression
lasso_model <- cv.glmnet(X_matrix, y_vector, alpha = 1, family = "gaussian", standardize = TRUE)
lasso_coef <- coef(lasso_model, s = "lambda.min")

# Elastic Net Regression
enet_model <- cv.glmnet(X_matrix, y_vector, alpha = 0.5, family = "gaussian", standardize = TRUE)
enet_coef <- coef(enet_model, s = "lambda.min")

# Poisson GLM
poisson_model <- glm(cnt ~ ., data = data %>% select(-instant, -dteday, -casual, -registered, -yr,
                                                     -atemp), family = poisson())
poisson_coef <- coef(poisson_model)
summary(poisson_model)

# Combine results into a table
coef_table <- data.frame(
  Variable = rownames(lasso_coef),
  Lasso = as.numeric(lasso_coef),
  ElasticNet = as.numeric(enet_coef),
  Poisson_GLM = poisson_coef[match(rownames(lasso_coef), names(poisson_coef))]
)

# Remove intercept row for clean display
coef_table <- coef_table[coef_table$Variable != "(Intercept)", ]

# Show result
print(coef_table)


# Feature Importance
library(randomForest)

# Train a Random Forest model

rf_model <- randomForest(cnt ~ ., data = data %>% select(-instant, -dteday, -casual, -registered,
                                                         -yr, -atemp), importance = TRUE, ntree = 500)


print(importance(rf_model))
varImpPlot(rf_model, main = "Random Forest Variable Importance")


# Load necessary libraries
library(xgboost)
library(Matrix)
library(dplyr)
library(ggplot2)

# Prepare the data
target <- data$cnt
df_matrix <- sparse.model.matrix(cnt ~ . -1, data = data %>% select(-instant, -dteday, -casual, -registered,
                                                                    -yr, -atemp))

# Convert to DMatrix format
dtrain <- xgb.DMatrix(data = df_matrix, label = target)

# Train XGBoost model with Poisson regression
xgb_model <- xgboost(data = dtrain,
                     max_depth = 6,
                     eta = 0.1,
                     nrounds = 200,
                     objective = "count:poisson",  # <-- use Poisson regression objective
                     eval_metric = "poisson-nloglik",  # optional evaluation metric
                     verbose = 0)

# Get feature importance
importance_matrix <- xgb.importance(feature_names = colnames(df_matrix), model = xgb_model)

# Print feature importance
print(importance_matrix)

# Plot feature importance
ggplot(importance_matrix, aes(x = reorder(Feature, Gain), y = Gain, fill = Gain)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "XGBoost Feature Importance (Poisson Gain)", x = "Variable", y = "Gain") +
  theme_minimal() +
  scale_fill_gradient(low = "blue", high = "red")


library(rstanarm)

# Fit Bayesian regression with Normal prior
bayesian_normal <- stan_glm(cnt ~ ., family = poisson, prior = normal(), 
                            data = data %>% select(-instant, -dteday, -casual, -registered,
                                                                  -yr, -atemp ))
summary(bayesian_normal)

# Plot 95% credible intervals
plot(bayesian_normal, prob = 0.95)


# Fit Bayesian regression with Laplace prior
bayesian_laplace <- stan_glm(cnt ~ ., family = poisson, prior = laplace(), 
                          data = data %>% select(-instant, -dteday, -casual, -registered,-yr, -atemp))
summary(bayesian_laplace)

# Plot 95% credible intervals
plot(bayesian_laplace, prob = 0.9)



library(dplyr)
library(tibble)
library(rstanarm)
library(ggplot2)

# Step 1: Extract Posterior Summary
summary_normal <- summary(bayesian_normal)
summary_laplace <- summary(bayesian_laplace)

# 95% Credible Intervals
ci_normal <- as.data.frame(posterior_interval(bayesian_normal, prob = 0.95)) %>%
  rownames_to_column("predictor")
ci_laplace <- as.data.frame(posterior_interval(bayesian_laplace, prob = 0.95)) %>%
  rownames_to_column("predictor")

# Add posterior mean from summary()
coef_normal <- summary_normal %>%
  as.data.frame() %>%
  rownames_to_column("predictor") %>%
  select(predictor, mean)

coef_laplace <- summary_laplace %>%
  as.data.frame() %>%
  rownames_to_column("predictor") %>%
  select(predictor, mean)

# Merge means with intervals
coef_normal <- left_join(coef_normal, ci_normal, by = "predictor") %>%
  mutate(prior = "Normal")

coef_laplace <- left_join(coef_laplace, ci_laplace, by = "predictor") %>%
  mutate(prior = "Laplace")

# Combine both
coef_compare <- bind_rows(coef_normal, coef_laplace) %>%
  rename(lower = `2.5%`, upper = `97.5%`) %>%
  mutate(significant = ifelse(lower > 0 | upper < 0, "Significant", "Not Significant"))

# Optional: Rename predictors for better display (edit as needed)
rename_preds <- c(
  "season" = "Season",
  "mnth" = "Month",
  "holiday" = "Holiday",
  "weekday" = "Weekday",
  "workingday" = "Working Day",
  "weathersit" = "Weather Situation",
  "temp" = "Temperature",
  "hum" = "Humidity",
  "windspeed" = "Windspeed"
)

# Plot function
plot_bayes_ci <- function(prior_name, color_sig = "#1b7837", color_nonsig = "#2166ac") {
  coef_compare %>%
    filter(prior == prior_name) %>%
    mutate(
      display_name = recode(predictor, !!!rename_preds),
      display_name = factor(display_name, levels = rev(unique(display_name)))
    ) %>%
    ggplot(aes(x = mean, y = display_name, color = significant)) +
    geom_point(size = 2) +
    geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.2) +
    scale_color_manual(values = c("Significant" = color_sig, "Not Significant" = color_nonsig)) +
    geom_vline(xintercept = 0, linetype = "dashed") +
    labs(
      title = paste("Bayesian 95% Credible Intervals (", prior_name, " Prior)", sep = ""),
      x = "Posterior Mean with 95% CI", y = "Predictor", color = "Significance"
    ) +
    theme_minimal()
}

# Step 4: Plot both
plot_bayes_ci("Normal")
plot_bayes_ci("Laplace")
