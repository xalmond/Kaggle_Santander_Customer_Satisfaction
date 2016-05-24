# Necessary libraries

library(xgboost)
library(ROCR)
library(Ckmeans.1d.dp)

# Reading data and preliminar review

df_train <- read.csv('./data/train.csv', header = TRUE, sep = ',', na.strings = c("","NA"))
df_target <- df_train$TARGET
df_train$ID <- NULL
df_train$TARGET <- NULL

df_test <- read.csv('./data/test.csv', header = TRUE, sep = ',', na.strings = c("","NA"))
df_test_target <- df_test$TARGET
df_test$ID <- NULL
df_test$TARGET <- NULL

cat("\nTrain:  número de filas: ", dim(df_train)[1], ", número de columnas: ", dim(df_train)[2],
    "\n\nTest:   número de filas: ", dim(df_train)[1], ", número de columnas: ", dim(df_train)[2],
    "\n\nTARGET: número de 0's:   ", length(df_target)-sum(df_target), ", número de 1's:     ", sum(df_target),
    "\n\n")

# Managing Outliers 

head(sort(unique(df_train$var3)))
df_train[df_train$var3 == -999999,c("var3")] <- -1
head(sort(unique(df_train$var3)))

head(sort(unique(df_test$var3)))
df_test[df_test$var3 == -999999,c("var3")] <- -1
head(sort(unique(df_test$var3)))

# Adding new feature

df_train$num_zeros <- rowSums(df_train == 0)
df_test$num_zeros <- rowSums(df_test == 0)

# Low Variance Filter

var_threshold <- 0
names_prev <- names(df_train)
df_test <- df_test[,apply(df_train,2,var) != var_threshold]
df_train <- df_train[,apply(df_train,2,var) != var_threshold]
names_prev[!(names_prev %in% names(df_train))]
cat("Número de filas: ", dim(df_train)[1], " Número de columnas: ", dim(df_train)[2])

# High Correlation Filter

cor_threshold <- 0.999
names_prev <- names(df_train)
df_cor <- abs(cor(df_train))
diag(df_cor) <- 0
df_cor[lower.tri(df_cor)] <- 0
df_test <- df_test[,!(apply(df_cor,2,max) >= cor_threshold)]
df_train <- df_train[,!(apply(df_cor,2,max) >= cor_threshold)]
names_prev[!(names_prev %in% names(df_train))]
cat("Número de filas: ", dim(df_train)[1], " Número de columnas: ", dim(df_train)[2])

# XGBoost Feature Importance

set.seed(1967)
names_prev <- names(df_train)
matrix_train <- xgb.DMatrix(data.matrix(df_train), label = as.numeric(df_target))
fix_params <- list(booster = "gbtree", 
                   objective = "binary:logistic", 
                   eval_metric = "auc")
model <- xgb.train(data = matrix_train, param = fix_params, nrounds = 100, verbose = 0)
imp_matrix <- xgb.importance(feature_names = names(df_train), model = model)
df_test <- df_test[,imp_matrix[imp_matrix$Gain>0.001]$Feature]
df_train <- df_train[,imp_matrix[imp_matrix$Gain>0.001]$Feature]
names_prev[!(names_prev %in% names(df_train))]
cat("Número de filas: ", dim(df_train)[1], " Número de columnas: ", dim(df_train)[2])

xgb.plot.importance(imp_matrix[imp_matrix$Gain>0.001])

# XGBoost, default parameters, training vs test comparison

matrix_train <- xgb.DMatrix(data.matrix(df_train), label = as.numeric(df_target))
model_cv <- xgb.cv(data = matrix_train, param = fix_params, nrounds = 100, verbose = 0, nfold = 10)

max_round <- which.max(model_cv$test.auc.mean)
plot( model_cv$train.auc.mean, type="l", lwd=2, col="red" , ylim=c(0.70, 1),
      main="AUC - train vs test" , xlab="nrounds", ylab="AUC")
lines(model_cv$test.auc.mean, type="l", lwd=2, col="blue" )
abline(v=max_round, lty=4)
legend(76,0.77,c("train","test"),lty=c(1,1),col=c("red","blue"))
text(42,0.855,paste("max nround  = ", max_round))

# Cross-validation process

cv_done <- TRUE
if (cv_done){
  all_lines <- read.csv("./data/result_cv.csv", header = TRUE, sep = ",")
} else {
  all_lines <- NULL
  for (n in 0:150){
    set.seed(1000+n)
    tree_params <- list(eta = runif(1, 0.010, 0.04),
                        max_depth = sample(5:8, 1),
                        max_delta_step = sample(0:3, 1),
                        subsample = runif(1, 0.7, 0.99),
                        colsample_bytree = runif(1, 0.5, 0.99))
    model_cv <- xgb.cv(param = append(fix_params,tree_params),
                       data = matrix_train,
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
    all_lines <- rbind(all_lines, new_line)
  }
  write.csv(all_lines,"./data/result_cv.csv", row.names = FALSE)
}
  
# XGBoost training fit using best parameters

best <- all_lines[which.max(all_lines$best_auc),]
tree_params <- list(eta = best$eta,
                    max_depth = best$max_depth,
                    max_delta_step = best$max_delta_step,
                    subsample = best$subsample,
                    colsample_bytree = best$colsample_bytree)

cat(paste("Parámetros óptimos:",
          "\n   eta              =", round(best$eta,2),
          "\n   max_depth        =", best$max_depth,
          "\n   max_delta_step   =", best$max_delta_step,
          "\n   subsample        =", round(best$subsample,2),
          "\n   colsample_bytree =", round(best$colsample_bytree,2)))


set.seed(1967)
ini_time <- proc.time()
model <- xgb.train(data = matrix_train, param = append(fix_params,tree_params), nrounds = best$best_itr, verbose = 0)
cat("Tiempo de entrenamiento: ", round((proc.time()-ini_time)[3]), "seg")

# Training Prediction

prob_train <- predict(model, matrix_train)
pred_train <- prediction(prob_train,df_target)
perf_train <- performance(pred_train, measure = "tpr", x.measure = "fpr")

# Test Prediction

prob_test <- predict(model, data.matrix(df_test))
pred_test <- prediction(prob_test,df_test_target)
perf_test <- performance(pred_test, measure = "tpr", x.measure = "fpr")

# ROC Curve

plot(x = perf_train@x.values[[1]], y = perf_train@y.values[[1]], col="red", lwd=2, type="l", 
     main="ROC Curve" , xlab="FPR or (1 - specificity)", ylab="TPR or sensitivity" )
lines(x = perf_test@x.values[[1]], y = perf_test@y.values[[1]], col="blue", lwd=2)
legend(.75,0.25,c("train","test"),lty=c(1,1),col=c("red","blue"))

# Training and Test AUC result

auc_perf <- performance(pred_train, measure = "auc")
auc_train_value <- auc_perf@y.values[[1]]
auc_perf <- performance(pred_test, measure = "auc")
auc_test_value <- auc_perf@y.values[[1]]
cat(paste0("Predicción Básica. Training AUC = ", round(auc_train_value,6)," Test AUC = ", round(auc_test_value,6)))


