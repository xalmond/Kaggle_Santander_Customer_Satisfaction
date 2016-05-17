# Necessary libraries

library(xgboost)
library(ROCR)

# Reading data and preliminar review

df_train <- read.csv('./data/train.csv', header = TRUE, sep = ',', na.strings = c("","NA"))
df_target <- df_train$TARGET
df_train$ID <- NULL
df_train$TARGET <- NULL
cat("\ntrain:  número de filas: ", dim(df_train)[1], ", número de columnas: ", dim(df_train)[2],
    "\n\nTARGET: número de 0's:   ", length(df_target)-sum(df_target), ", número de 1's:     ", sum(df_target),
    "\n\n")

# Low Variance Filter

var_threshold <- 0
names_prev <- names(df_train)
df_train <- df_train[,apply(df_train,2,var) != var_threshold]
names_prev[!(names_prev %in% names(df_train))]
cat("Número de filas: ", dim(df_train)[1], " Número de columnas: ", dim(df_train)[2])

# High Correlation Filter

cor_threshold <- 0.999
names_prev <- names(df_train)
df_cor <- abs(cor(df_train))
diag(df_cor) <- 0
df_cor[lower.tri(df_cor)] <- 0
df_train <- df_train[,!(apply(df_cor,2,max) >= cor_threshold)]
names_prev[!(names_prev %in% names(df_train))]
cat("Número de filas: ", dim(df_train)[1], " Número de columnas: ", dim(df_train)[2])

# XGBoost Feature Importance

names_prev <- names(df_train)
matrix_train <- xgb.DMatrix(data.matrix(df_train), label = as.numeric(df_target))
fix_params <- list(booster = "gbtree", 
                   objective = "binary:logistic", 
                   eval_metric = "auc")
model <- xgb.train(data = matrix_train, param = fix_params, nrounds = 100, verbose = 0)
imp_matrix <- xgb.importance(feature_names = names(df_train), model = model)
df_train <- df_train[,imp_matrix[imp_matrix$Gain>0.001]$Feature]
names_prev[!(names_prev %in% names(df_train))]
cat("Número de filas: ", dim(df_train)[1], " Número de columnas: ", dim(df_train)[2])

xgb.plot.importance(imp_matrix[imp_matrix$Gain>0.001])

# XGBoost, default parameters, training vs test comparison

matrix_train <- xgb.DMatrix(data.matrix(df_train), label = as.numeric(df_target))
set.seed(1967)
ini_time <- proc.time()
model.cv <- xgb.cv(data = matrix_train, param = fix_params, nrounds = 100, verbose = 0, nfold = 10)

max_round <- which.max(model.cv$test.auc.mean)
plot( model.cv$train.auc.mean, type="l", lwd=2, col="red" , ylim=c(0.80, 0.95),
      main="AUC - train vs test" , xlab="nrounds", ylab="AUC")
lines( model.cv$test.auc.mean, type="l", lwd=2, col="blue" )
legend(70,0.905,c("train","test"),lty=c(1,1),col=c("red","blue"))
abline(v=max_round, lty=4)
text(36,0.845,paste("max nround  = ", max_round))

# Cross-validation process

cv_done <- TRUE
if (cv_done){
  all_lines <- read.csv("./data/result_cv.csv", header = TRUE, sep = ",")
} else {
  all_lines <- NULL
  for (n in 85:150){
    set.seed(1000+n)
    tree_params <- list(eta = runif(1, 0.010, 0.04),
                        max_depth = sample(5:8, 1),
                        max_delta_step = sample(0:3, 1),
                        subsample = runif(1, 0.7, 0.99),
                        colsample_bytree = runif(1, 0.5, 0.99))
    model_cv <- xgb.cv(param = append(fix_params,tree_params),
                       data = xgb_data,
                       nrounds = 1e4,
                       nfold = 10,
                       early.stop.round = 100,
                       verbose = 0)
    new_line <- data.frame( eta = tree_params$eta,
                            max_depth = tree_params$max_depth,
                            max_delta_step = tree_params$max_delta_step,
                            subsample = tree_params$subsample,
                            colsample_bytree = tree_params$colsample_bytree,
                            best_itr = which.max(model_cv$test.auc.mean),
                            best_auc = max(model_cv$test.auc.mean))
    print(new_line)
    all_lines <- rbind(all_lines, new_line)
  }
}

# XGBoost training fit and prediction using best parameters

best <- all_lines[which.max(all_lines$best_auc),]
tree_params <- list(eta = best$eta,
                    max_depth = best$max_depth,
                    max_delta_step = best$max_delta_step,
                    subsample = best$subsample,
                    colsample_bytree = best$colsample_bytree)
set.seed(1967)
ini_time <- proc.time()
model <- xgb.train(data = matrix_train, param = append(fix_params,tree_params), nrounds = best$best_itr, verbose = 0)
pred_train <- predict(model, matrix_train)
cat("Tiempo de predicción: ", round((proc.time()-ini_time)[3]), "seg")

# ROC Curve and AUC result

pred <- prediction(pred_train,df_target)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, col="red", lwd=2, main="ROC Curve" , xlab="FPR or (1 - specificity)", ylab="TPR or sensitivity" )
auc_perf <- performance(pred, measure = "auc")
auc_value <- auc_perf@y.values[[1]]
cat("AUC para Predicción con cross-validation: ", auc_value)