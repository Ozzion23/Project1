library(leaps)
library(glmnet)
library(caret)
library(tidyverse)
library(corrplot)

# 1) Prepare Data
inforce <- read.csv("C:/Users/achar/Downloads/inforce10k.csv")
fmv <- read.csv("C:/Users/achar/Downloads/fmv_seriatim.csv")

inforce2 <- merge(inforce, fmv, by.x = "recordID", by.y = "RecordID")

vNames <- c("recordID", "gender", "prodType", "issueDate", "matDate", "age",
            "gmdbAmt",  "gmwbAmt", "gmwbBalance", "gmmbAmt", "withdrawal", 
            paste("FundValue", 1:10, sep = ""), "fmv")

dat10k <- inforce2[, vNames]

# 2) Explore data

summary(dat10k[1:22])
glimpse(dat10k[1:22])

dim(dat10k)
names(dat10k)
sum(is.na(dat10k))

table(dat10k$gender)
table(dat10k$prodType)

aggregate(dat10k$fmv, 
          by = list(dat10k$gender),
          FUN = mean)

aggregate(dat10k$fmv,
          by = list(dat10k$prodType),
          FUN = mean)


# Create correlation plot
cor_matrix <- cor(dat10k[, c("issueDate","matDate","age","gmdbAmt","gmwbAmt",
                             "gmwbBalance","gmmbAmt","withdrawal","FundValue1",
                             "FundValue2","FundValue3","FundValue4",
                             "FundValue5","FundValue6","FundValue7",
                             "FundValue8","FundValue9","FundValue10","fmv")])

col <- colorRampPalette(c("darkblue", "white", "darkred"))(200)

corrplot(cor_matrix, method = "color", type = "upper",
         col = col, tl.pos = 'lt', tl.col = 'black', 
         number.cex = 0.9, tl.cex = 0.8)


# 3) Split data into training and testing sets

set.seed(945)
train_index <- createDataPartition(dat10k$fmv, p = 0.8, list = FALSE)
train_data <- dat10k[train_index, ]
test_data <- dat10k[-train_index, ]


# 4) Linear, LASSO, Ridge (Supervised)

# First create dummy variables for training data
train_data$gender <- as.numeric(train_data$gender == "M")  # M = 1, F = 0
train_data <- cbind(train_data, model.matrix(~ prodType - 1, train_data)[,-1])

# First create dummy variables for training data
test_data$gender <- as.numeric(test_data$gender == "M")  # M = 1, F = 0
test_data <- cbind(test_data, model.matrix(~ prodType - 1, test_data)[,-1])

# Create X and y values for updated training set
X <- as.data.frame(train_data[, !(names(train_data) %in% c("recordID", "fmv"))])
y <- train_data$fmv

# Create X and y values for updated testing set
X_test <- as.data.frame(test_data[, !(names(test_data) 
                                      %in% c("recordID", "fmv"))])

y_test <- test_data$fmv

# Combine X and y into a single data frame (Train and Test)
train_data_combined <- as.data.frame(cbind(fmv = y, X))
train_data_combined <- subset(train_data_combined, select = -prodType)

test_data_combined <- as.data.frame(cbind(fmv = y_test, X_test))
test_data_combined <- subset(test_data_combined, select = -prodType)

# Use Exhaustive Approach to find the best model
reg_search <- regsubsets(fmv ~ ., data = train_data_combined, 
                         method = "exhaustive", 
                         nvmax = ncol(train_data_combined))

reg_summary <- summary(reg_search)
best_model_size <- which.min(reg_summary$bic)
best_vars <- names(coef(reg_search, best_model_size))[-1]
best_vars

plot(reg_search, scale = "bic")

formula1 <- as.formula(paste("fmv ~", paste(best_vars, collapse = " + ")))
lm_model1 <- lm(formula1, data = train_data_combined)
summary(lm_model1)

# Ridge Regression with CV
X_train <- model.matrix(~ . -fmv, data = train_data_combined)[,-1]
X_test <- model.matrix(~. -fmv, data = test_data_combined)[,-1]
y_train <- train_data$fmv
y_test <- test_data$fmv

ridge_cv <- cv.glmnet(X_train, y_train, alpha = 0)
ridge_cv$lambda.min
ridge_model <- glmnet(X_train, y_train, alpha=0, lambda=ridge_cv$lambda.min)
ridge_coefficients <- coef(ridge_model, s = ridge_cv$lambda.min)
print(ridge_coefficients)

# Lasso Regression with CV
lasso_cv <- cv.glmnet(X_train, y_train, alpha = 1)
lasso_cv$lambda.min
lasso_model <- glmnet(X_train, y_train, alpha = 1, lambda = lasso_cv$lambda.min)
lasso_coefficients <- coef(lasso_model, s = lasso_cv$lambda.min)
print(lasso_coefficients)

# 5) Hierarchical Clustering (Linear, Lasso, Ridge)

# Perform clustering
dist_matrix <- dist(scale(X_train))
hclust_result <- hclust(dist_matrix)
clusters_h <- cutree(hclust_result, k = 200)

# Select one observation from each cluster
cluster_samples <- sapply(1:200, function(i) {
  cluster_indices <- which(clusters_h == i)
  sample(cluster_indices, 1)
})

# Get data for selected samples 
train_selected_h <- train_data_combined[cluster_samples, ]
X_train_selected_h <- X_train[cluster_samples, ]
y_train_selected_h <- y_train[cluster_samples]


# Linear regression with selected samples using Hierarchical clustering
reg_search_h <- regsubsets(fmv ~., data = train_selected_h, method = "exhaustive"
                           , nvmax = ncol(train_selected_h))

plot(reg_search_h, scale = "bic")

reg_summary_h <- summary(reg_search_h)
best_model_size_h <- which.min(reg_summary_h$bic)
best_vars_h <- names(coef(reg_search_h, best_model_size_h))[-1]
best_vars_h

formula2 <- as.formula(paste("fmv ~", paste(best_vars_h, collapse = " + ")))
lm_model2 <- lm(formula2, data = train_selected_h)
summary(lm_model2)

# Ridge Regression with selected samples from Hierarchical clustering
ridge_cv_h <- cv.glmnet(X_train_selected_h, y_train_selected_h, alpha = 0)
ridge_cv_h$lambda.min
ridge_model_h <- glmnet(X_train_selected_h, y_train_selected_h, alpha = 0,
                        lambda = ridge_cv_h$lambda.min)
ridge_coefficients_h <- coef(ridge_model_h, s = ridge_cv_h$lambda.min)
print(ridge_coefficients_h)

# Lasso Regression with selected samples from Hierarchical clustering
lasso_cv_h <- cv.glmnet(X_train_selected_h, y_train_selected_h, alpha = 1)
lasso_cv_h$lambda.min
lasso_model_h <- glmnet(X_train_selected_h, y_train_selected_h, alpha = 1,
                        lambda = lasso_cv_h$lambda.min)
lasso_coefficients_h <- coef(lasso_model_h, s = lasso_cv_h$lambda.min)
print(lasso_coefficients_h)


# 6) K-means clustering (Linear, Lasso, Ridge)

set.seed(945)
kmean_result <- kmeans(scale(X_train), centers = 200)

cluster_samples_k <- sapply(1:200, function(i) {
  cluster_indices <- which(kmean_result$cluster == i)
  if(length(cluster_indices) > 0) {
    sample(cluster_indices, 1)
  } else {
    NA # Handle any empty clusters if they do happen to pop up
  }
})

# Remove any NA values if clusters that were empty appeared
cluster_samples_k <- cluster_samples_k[!is.na(cluster_samples_k)]

# Get data for selected samples from K-means clustering
train_selected_k <- train_data_combined[cluster_samples_k, ]
X_train_selected_k <- X_train[cluster_samples_k, ]
y_train_selected_k <- y_train[cluster_samples_k]

# Linear Regression using K-means cluster
reg_search_k <- regsubsets(fmv~., data = train_selected_k, 
                           method = "exhaustive", 
                           nvmax = ncol(train_selected_k))

plot(reg_search_k, scale = "bic")

reg_summary_k <- summary(reg_search_k)
best_model_size_k <- which.min(reg_summary_k$bic)
best_vars_k <- names(coef(reg_search_k, best_model_size_k))[-1]
best_vars_k

formula3 <- as.formula(paste("fmv ~", paste(best_vars_k, collapse = " + ")))
lm_model3 <- lm(formula3, data = train_selected_k)
summary(lm_model3)

# Ridge Regression with K-means clustering
ridge_cv_k <- cv.glmnet(X_train_selected_k, y_train_selected_k, alpha = 0)
ridge_cv_k$lambda.min
ridge_model_k <- glmnet(X_train_selected_k, y_train_selected_k, alpha = 0,
                        lambda = ridge_cv_k$lambda.min)
ridge_coefficients_k <- coef(ridge_model_k, s = ridge_cv_k$lambda.min)
print(ridge_coefficients_k)


# LASSO Regression with K-means clustering
lasso_cv_k <- cv.glmnet(X_train_selected_k, y_train_selected_k, alpha = 1)
lasso_cv_k$lambda.min
lasso_model_k <- glmnet(X_train_selected_k, y_train_selected_k, alpha = 1,
                        lambda = lasso_cv_k$lambda.min)
lasso_coefficients_k <- coef(lasso_model_k, s = lasso_cv_k$lambda.min)
print(lasso_coefficients_k)


# Make prediction for all 9 models
predictions <- list(
  
  # Original Models (Step 4)
  lm_predictions = predict(lm_model1, newdata = test_data_combined),
  ridge_predictions = predict(ridge_model, s = ridge_cv$lambda.min, 
                               newx = X_test),
  lasso_predictions = predict(lasso_model, s = lasso_cv$lambda.min, 
                               newx = X_test),
  
  # Hierarchical clustering models (Step 5)
  lm_h = predict(lm_model2, newdata = test_data_combined),
  ridge_h = predict(ridge_model_h, s = ridge_cv_h$lambda.min, newx = X_test),
  lasso_h = predict(lasso_model_h, s = lasso_cv_h$lambda.min, newx = X_test),
  
  # K-means clustering models (Step 6)
  lm_k = predict(lm_model3, newdata = test_data_combined),
  ridge_k = predict(ridge_model_k, s = ridge_cv_k$lambda.min, newx = X_test),
  lasso_k = predict(lasso_model_k, s = lasso_cv_k$lambda.min, newx = X_test)
)

# # Calculate prediction errors for all models
mse_results <- sapply(predictions, function(pred) mean(y_test - pred)^2)
rmse_results <- sqrt(mse_results)

# Table summary of prediction errors
test_error_summary <- data.frame(
  Model = names(mse_results),
  MSE = mse_results,
  RMSE = rmse_results,
  row.names = NULL
)

print("\nTest Error Summary:")
print(test_error_summary)

# 7) Calculate VaR for all predictions and observed values

# Function to calculate VaR
calculate_var <- function(values, levels = c(0.90, 0.95)) {
  sapply(levels, function(level) quantile(values, level))
}

var_results <- lapply(predictions, calculate_var)
observed_var <- calculate_var(y_test)

# Create summary table of results
var_summary <- data.frame(
  Model = names(var_results),
  VaR_90 = sapply(var_results, function(x) x[1]),
  VaR_95 = sapply(var_results, function(x) x[2]),
  row.names = NULL
)

# Observed VaR
var_summary <- rbind(var_summary,
                     data.frame(Model = "Observed",
                                VaR_90 = observed_var[1],
                                VaR_95 = observed_var[2]),
                     row.names = NULL)

# Print results
print("Value at Risk Summary:")
print(var_summary)

