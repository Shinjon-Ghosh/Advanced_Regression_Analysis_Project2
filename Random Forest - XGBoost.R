setwd("F:/Applied Statistics Course - ISU/Advanced Regression Analysis/Project/Project 2")

library(randomForest)
library(xgboost)
library(dplyr)
library(ggplot2)

`%nin%` <- Negate(`%in%`)

# Define the true variable indices for each target
true_vars_list <- list(
  y1 = c(1, 20, 40, 60, 80, 90),
  y2 = c(1, 20, 40, 60, 80, 90),
  y3 = c(1, 20, 40, 60, 80, 90)
)

data <- read.csv(file = "Data/simulated_poisson_6000rows.csv")

results_all <- data.frame()

# Loop through 30 datasets per each of y1, y2, y3 (total: 90)
for (i in 0:29) {
  for (target in c("y1", "y2", "y3")) {
    
    dataset_label <- paste0(target, "_dataset", i + 1)
    start_idx <- i * 200 + 1
    end_idx <- start_idx + 99
    
    y <- data[[target]][start_idx:end_idx]
    X <- data[start_idx:end_idx, 4:103]  # x1 to x100
    
    # ===== RANDOM FOREST =====
    rf_model <- randomForest(x = X, y = y, ntree = 500, importance = TRUE)
    imp_rf <- importance(rf_model)[, 1]  # %IncMSE
    selected_rf <- as.integer(sub("x", "", names(sort(imp_rf, decreasing = TRUE)[1:10])))
    
    # Evaluation
    true_vars <- true_vars_list[[target]]
    TP_rf <- sum(true_vars %in% selected_rf)
    FP_rf <- sum(!(1:100 %in% true_vars) & (1:100 %in% selected_rf))
    FN_rf <- sum(true_vars %nin% selected_rf)
    TN_rf <- sum(!(1:100 %in% true_vars) & !(1:100 %in% selected_rf))
    TPR_rf <- if ((TP_rf + FN_rf) == 0) NA else TP_rf / (TP_rf + FN_rf)
    TNR_rf <- if ((TN_rf + FP_rf) == 0) NA else TN_rf / (TN_rf + FP_rf)
    
    results_all <- rbind(results_all, data.frame(
      response = target,
      dataset = i + 1,
      model = "RandomForest",
      TP = TP_rf, FP = FP_rf, FN = FN_rf, TN = TN_rf,
      TPR = TPR_rf, TNR = TNR_rf
    ))
    
    # ===== XGBOOST =====
    dtrain <- xgb.DMatrix(data = as.matrix(X), label = y)
    xgb_model <- xgboost(data = dtrain, objective = "reg:squarederror", nrounds = 100, verbose = 0)
    imp_xgb <- xgb.importance(model = xgb_model)
    
    selected_xgb <- as.integer(sub("x", "", imp_xgb$Feature[1:10]))
    
    TP_xgb <- sum(true_vars %in% selected_xgb)
    FP_xgb <- sum(!(1:100 %in% true_vars) & (1:100 %in% selected_xgb))
    FN_xgb <- sum(true_vars %nin% selected_xgb)
    TN_xgb <- sum(!(1:100 %in% true_vars) & !(1:100 %in% selected_xgb))
    TPR_xgb <- if ((TP_xgb + FN_xgb) == 0) NA else TP_xgb / (TP_xgb + FN_xgb)
    TNR_xgb <- if ((TN_xgb + FP_xgb) == 0) NA else TN_xgb / (TN_xgb + FP_xgb)
    
    results_all <- rbind(results_all, data.frame(
      response = target,
      dataset = i + 1,
      model = "XGBoost",
      TP = TP_xgb, FP = FP_xgb, FN = FN_xgb, TN = TN_xgb,
      TPR = TPR_xgb, TNR = TNR_xgb
    ))
  }
}


results_all_summary <- results_all %>%
  group_by(response, model) %>%
  summarise(across(c(TP, FP, FN, TN, TPR, TNR), mean), .groups = "drop")

print(results_all_summary)

# Boxplots
# pdf(file ="Figure/box_TPR_RF_XGB.pdf")
# op<-par(mar=c(5,5,4,1),cex.axis=1.5,font.axis=2,font.lab=2,cex.main=2,cex.lab=1.6)
ggplot(results_all, aes(x = model, y = TPR, fill = model)) +
  geom_boxplot() +
  facet_wrap(~ response) +
  #labs(title = "True Positive Rate by Response", y = "TPR") +
  theme_minimal()+
  theme(legend.position = "none")
# par(op)
# dev.off()

# pdf(file ="Figure/box_TNR_RF_XGB.pdf")
# op<-par(mar=c(5,5,4,1),cex.axis=1.5,font.axis=2,font.lab=2,cex.main=2,cex.lab=1.6)
ggplot(results_all, aes(x = model, y = TNR, fill = model)) +
  geom_boxplot() +
  facet_wrap(~ response) +
  #labs(title = "True Negative Rate by Response", y = "TNR") +
  theme_minimal()+
  theme(legend.position = "none")
# par(op)
# dev.off()
# 
library(reshape2)
library(ggplot2)

# Initialize frequency storage: [variable, response, model]
var_freq <- array(0, dim = c(100, 3, 2), 
                  dimnames = list(
                    paste0("x", 1:100),
                    c("y1", "y2", "y3"),
                    c("RandomForest", "XGBoost")
                  ))

# Re-run selection collection only (no evaluation)
for (i in 0:29) {
  for (target in c("y1", "y2", "y3")) {
    start_idx <- i * 200 + 1
    end_idx <- start_idx + 99
    y <- data[[target]][start_idx:end_idx]
    X <- data[start_idx:end_idx, 4:103]
    
    # Random Forest
    rf_model <- randomForest(x = X, y = y, ntree = 500, importance = TRUE)
    imp_rf <- importance(rf_model)[, 1]
    selected_rf <- names(sort(imp_rf, decreasing = TRUE))[1:10]
    var_freq[selected_rf, target, "RandomForest"] <- var_freq[selected_rf, target, "RandomForest"] + 1
    
    # XGBoost
    dtrain <- xgb.DMatrix(data = as.matrix(X), label = y)
    xgb_model <- xgboost(data = dtrain, objective = "reg:squarederror", nrounds = 100, verbose = 0)
    
    imp_xgb <- xgb.importance(model = xgb_model)
    selected_xgb <- imp_xgb$Feature[1:min(10, nrow(imp_xgb))]  # take top 10 or fewer if not enough
    
    # Keep only features that are present in var_freq row names
    selected_xgb_valid <- selected_xgb[selected_xgb %in% rownames(var_freq)]
    var_freq[selected_xgb_valid, target, "XGBoost"] <- var_freq[selected_xgb_valid, target, "XGBoost"] + 1
    
    
  }
}

# Melt for heatmap
df_heat <- melt(var_freq)
colnames(df_heat) <- c("Variable", "Response", "Model", "Frequency")

df_heat_all <- df_heat

df_heat_all$Variable <- factor(df_heat_all$Variable, levels = paste0("x", 1:100))


# pdf(file ="Figure/VB_Freq_RF_XGB.pdf")
# op<-par(mar=c(5,5,4,1),cex.axis=1.5,font.axis=2,font.lab=2,cex.main=2,cex.lab=1.6)
ggplot(df_heat_all, aes(x = Model, y = Variable, fill = Frequency)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "red") +
  facet_wrap(~ Response) +
  scale_y_discrete(limits = rev(levels(df_heat_all$Variable))) + # Reverse the y-axis
  labs(
    title = "Selection Frequency of All Variables Across 30 Datasets",
    x = "Model",
    y = "Variable",
    fill = "Frequency"
  ) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 6))
# par(op)
# dev.off()




