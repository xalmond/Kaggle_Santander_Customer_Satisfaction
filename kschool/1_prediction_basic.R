# Necessary libraries

library(xgboost)
library(ROCR)

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

# Data Preparation

matrix_train <- xgb.DMatrix(data.matrix(df_train), label = as.numeric(df_target))
fix_params <- list(booster = "gbtree", 
                   objective = "binary:logistic", 
                   eval_metric = "auc")

# XGBoost training fit

set.seed(1967)
ini_time <- proc.time()
model <- xgb.train(data = matrix_train, param = fix_params, nrounds = 100, verbose = 0)
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


