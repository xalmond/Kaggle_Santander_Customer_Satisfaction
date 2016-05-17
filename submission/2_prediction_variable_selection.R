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

# XGBoost training fit and prediction

matrix_train <- xgb.DMatrix(data.matrix(df_train), label = as.numeric(df_target))
set.seed(1967)
ini_time <- proc.time()
model <- xgb.train(data = matrix_train, param = fix_params, nrounds = 100, verbose = 0)
pred_train <- predict(model, matrix_train)
cat("Tiempo de predicción: ", round((proc.time()-ini_time)[3]), "seg")

# ROC Curve and AUC result

pred <- prediction(pred_train,df_target)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, col="red", lwd=2, main="ROC Curve" , xlab="FPR or (1 - specificity)", ylab="TPR or sensitivity" )
auc_perf <- performance(pred, measure = "auc")
auc_value <- auc_perf@y.values[[1]]
cat("AUC para Predicción con selección de variables: ", auc_value)