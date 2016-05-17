
# Reading data and preliminar review

df_train_orig <- read.csv('./data/train_original.csv', header = TRUE, sep = ',', na.strings = c("","NA"))

set.seed(1967)

df_train_orig_zeros <- df_train_orig[df_train_orig$TARGET == 0,]
num_zeros <- nrow(df_train_orig_zeros)
sample_training <- sample(num_zeros,num_zeros/2)
df_train_zeros <- df_train_orig_zeros[sample_training,]
df_test_zeros <- df_train_orig_zeros[-sample_training,]

df_train_orig_ones <- df_train_orig[df_train_orig$TARGET == 1,]
num_ones <- nrow(df_train_orig_ones)
sample_training <- sample(num_ones,num_ones/2)
df_train_ones <- df_train_orig_ones[sample_training,]
df_test_ones <- df_train_orig_ones[-sample_training,]

df_train <- rbind(df_train_ones,df_train_zeros)
df_train <- df_train[order(df_train$ID),]

df_test <- rbind(df_test_ones,df_test_zeros)
df_test <- df_test[order(df_test$ID),]

write.csv(df_train,"./data/train.csv", row.names = FALSE)
write.csv(df_test,"./data/test.csv", row.names = FALSE)
