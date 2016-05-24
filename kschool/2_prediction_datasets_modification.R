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

# XGBoost training fit

matrix_train <- xgb.DMatrix(data.matrix(df_train), label = as.numeric(df_target))
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
cat(paste0("Predicción con selección de variables. Training AUC = ", round(auc_train_value,6),
           " Test AUC = ", round(auc_test_value,6)))


