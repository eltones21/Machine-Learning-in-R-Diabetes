#Installing the required R packages
if(!require(tidyverse))    install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret))        install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggthemes))     install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(ggcorrplot))   install.packages("ggcorrplot", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(magrittr))     install.packages("magrittr", repos = "http://cran.us.r-project.org")
if(!require(rpart))        install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot))   install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(neighbr))      install.packages("neighbr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(ggthemes)
library(ggcorrplot)
library(randomForest)
library(magrittr)
library(rpart)
library(rpart.plot)
library(neighbr)


#######----Data pre-processing------

#download database from UCI Machine Learning Repository
dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv", dl)

#Assign csv database into the data
data <- read_csv(dl, col_types = "dffffffffffffffff")
colnames(data) <- make.names(colnames(data))

# reorder the "yes-no" factors so that all are in the same order
# when reading in the data, if the first entry of a column is "No" for example, that is taken to be the first level
# this loop changes the order of the factors for which the first entry is "No"
no_ind <- which(data[1,]=="No")
for (i in no_ind) {
  data[,i] <- factor(data[[i]], levels = c("Yes","No"))
}

# create a validation set - this is used to assess the final model
# diabetes data set is used for model training and selection
set.seed(4)
validation_index <- createDataPartition(data$class, times=1, p=0.15, list=FALSE)
validation <- data[validation_index,]
diabetes <- data[-validation_index,]


######_______Training and Testing of Algorithm__________

# create train and test sets from diabetes. 
# train is used to construct various models and test is used to assess their performances
# the best performing model will then be retrained using the diabetes data set and assessed using the validation data set
set.seed(16)
test_index <- createDataPartition(diabetes$class, times=1, p=0.15, list=FALSE)
test <- diabetes[test_index,]
train <- diabetes[-test_index,]

rm(test_index, validation_index)


######-------Exploratory analysis----------------------

# set global theme (google docs)
theme_set(theme_gdocs())

#size of each database
nrow(diabetes)
nrow(train)
nrow(test)
nrow(validation)


# lets first visualize the correlation between the features (non significant correlations left blank)
# probably not worth using pca because the data set is quite small
bin_diabetes <- sapply({diabetes[,c(-1,-2,-17)] == "Yes"} %>% as_tibble, as.numeric) %>% 
                as_tibble %>% mutate(Gender = as.numeric(diabetes$Gender=="Male"))
correlation_matrix <- round(cor(bin_diabetes),1)

ggcorrplot(correlation_matrix, 
           method = "circle",
           insig = "blank",
           type = "lower",
           title = "Correlation of Diabetes variables", 
           legend.title = "Correlation",
           ggtheme = theme_gdocs())  
ggsave("rmd_files/images/correlation.png", width = 8, height = 6)

# compare genders. not many non-diabetic patients are female in the diabetes database
diabetes %>%
  ggplot(aes(class, fill = Gender))+
  geom_bar(width = 0.6, position = position_dodge(width = 0.7))+
  ylab("Number of Patients")+
  ggtitle("Number of Diabetic and Non-Diabet Patients by Gender")
ggsave("rmd_files/images/gender.png", width = 8, height = 5)

# comparing age. there does not seem to be a significant difference
diabetes %>%
  ggplot(aes(Age, class, fill = class))+
  geom_violin(alpha = 0.8)+
  ggtitle("Prevalence of Diabetes by Age")
ggsave("rmd_files/images/age.png", width = 8, height = 5)

#analysis of diabetes by obesity
#different than the general research, this database suggests obesity  
# is not a significant diabetes factor
diabetes %>%
  ggplot(aes(class, fill = Obesity))+
  geom_bar(width = 0.6, position = position_dodge(width = 0.7))+
  ylab("Number of Patients")+
  ggtitle("Diabetes by Obesity")
ggsave("rmd_files/images/obesity.png", width = 8, height = 5)

#based on the cor matrix it seems polydipsia and polyuria appear to be important
#if patient has both conditions, they are likely to be diabetic
diabetes %>%
  ggplot(aes(Polyuria, Polydipsia, color = class))+
  geom_jitter(height = 0.2, width = 0.2)+
  ggtitle("Prevalence of Diabetes by Polyuria and Polydipsia ")
ggsave("rmd_files/images/dipsiauria.png", width = 8, height = 5)

# this plot suggests that weakness might not be a significant feature
# however, weight loss could be significant
diabetes %>%
  ggplot(aes(sudden.weight.loss, weakness, colour = class)) +
  geom_jitter(height = 0.2, width = 0.2) +
  xlab("Sudden Weight Loss") +
  ylab("Weakness") +
  ggtitle("Prevalence of Diabetes by Sudden Weight Loss and Weakness")
ggsave("rmd_files/images/weightweak.png", width = 8, height = 5)



######____________Model 1 logistic regression__________

# now use the entire train data set and evaluate the model against the test set
# this section constructs a logistic regression model
model_glm <- glm(as.numeric(class=="Positive")~., family = "binomial", data = train)
preds_glm <- predict(model_glm, test, type = "response")                                                  
preds_glm <- ifelse(preds_glm>0.5, "Positive","Negative") %>% factor(levels = c("Positive","Negative")) 

# confusion matrix using test database
cm_glm <- confusionMatrix(preds_glm, test$class)

# save accuracy and sensitivity in the results tibble that will be our model summary table
acc_glm <- cm_glm$overall["Accuracy"]
sen_glm <- cm_glm$byClass["Sensitivity"]

results <- tibble(Method = "Model (1) Logistical Regression", Accuracy = acc_glm , Sensitivity = sen_glm )
results %>% knitr::kable()


######____________Model 2 Knn _________
# this section constructs a knn model

# the train data set is modified so that the features are logical
# a column ID is also required to run the knn function
train_knn <- {train[-c(1, 2, 17)]=="Yes"} %>% as_tibble # remove age, gender and class. then, convert to logical
train_knn <- cbind(train[2] == "Male", train_knn)       # add logical variable for gender
train_knn <- cbind(train_knn, train[17])                # add the class back in (it doesn't need to be logical, only the features)


# the test set needs to take the same format
# however, the ID and class columns need removed (requirement of knn function)
test_knn <- {test[-c(1, 2, 17)]=="Yes"} %>% as_tibble
test_knn <- cbind(test[2] == "Male", test_knn)

set.seed(4)

# here, 8-fold cross-validation is used to select an optimal k
f = seq(2, 8, 1)

# k-nearest neighbors model trained 7*18 = 126 times
#we try out 8 k from 1 to 8
knn_results <- sapply(f, function(f){
  control <- trainControl(method = "cv", number = f, p = .9)
  train_knn_cv <- train(class ~ ., method = "knn", 
                        data = train_knn,
                        tuneGrid = data.frame(k = seq(2, 19, 1)),
                        trControl = control)
  return(max(train_knn_cv$results$Accuracy))
})

# define the optimal k
n_folds <- which.max(knn_results) + 1

# retrain the knn model
control <- trainControl(method = "cv", number = n_folds, p = .9)
mod_knn <- train(class ~ ., method = "knn", 
                 data = train_knn,
                 tuneGrid = data.frame(k = seq(2, 19, 1)),
                 trControl = control)

#generate chart
ggplot(mod_knn, highlight = TRUE)

# generate the predictable results using test dataset
preds_knn <- predict(mod_knn, test_knn)

# confusion knn matrix
cm_knn <- confusionMatrix(preds_knn, test$class)

# save accuracy and sensitivity knn
acc_knn <- cm_knn$overall["Accuracy"]
sen_knn <- cm_knn$byClass["Sensitivity"]

# store knn results
results <- bind_rows(results, tibble(Method = "Model (2) Knn neighrbours", Accuracy = acc_knn , Sensitivity = sen_knn))
results %>% knitr::kable()


######____________Model 3 decision tree _________

# this section constructs a decision tree
# this is one the easiest model to interpret, it's easy to visualise how it makes decisions

# the train function in the caret package is used to select an optimal complexity parameter (cp) 
#using 25 bootstrap samples with replacement

set.seed(64)
model_tree <- train(class~.,
                    method = "rpart",
                    tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)),
                    trControl = trainControl(method = "cv", number=5, p=.9),
                    data = train)

opt_co <- as.numeric(model_tree$bestTune)

# visualize the performance of each cp
ggplot(model_tree, highlight = TRUE)

# plot the model - this helps to understand how the algorithm works
rpart.plot(model_tree$finalModel, box.palette="RdBu", shadow.col="gray", nn=TRUE)
title("Decision Tree")
  
# obtain predictions using opt_p
preds_tree <- predict(model_tree, test) 

# confusion matrix
cm_tree <- confusionMatrix(preds_tree, test$class)

# save accuracy and sensitivity
acc_tree <- cm_tree$overall["Accuracy"]
sen_tree <- cm_tree$byClass["Sensitivity"]

results <- bind_rows(results, tibble(Method = "Model (3) Decision tree", Accuracy = acc_tree , Sensitivity = sen_tree ))
results %>% knitr::kable()



######____________Model 4 Random forest _________

# this section expands on the idea of decision trees by creating a random forest
# a random forest is a collection of decision trees
# predictions are made using the majority votes from each tree
# to reduce the dependency between trees, a random subset of features can be chosen 
#at each node to decide on which split to make (if any). this is `mtry`

set.seed(11)
model_rf <- train(class~.,
                  method = "rf",
                  tuneGrid = data.frame(mtry = 3:11),
                  data = train)

#visualize the performance of each mtry
ggplot(model_rf, highlight = TRUE)+
  ggtitle("Accuracy for each number of randomly selected predictors")
ggsave("rmd_files/images/mtry.png", width = 8, height = 5)

# again, note the variability of each mtry
model_rf$results %>% 
  ggplot(aes(x = mtry, y = Accuracy)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(x = mtry, 
                    ymin = Accuracy - AccuracySD,
                    ymax = Accuracy + AccuracySD))

#generate predictions
preds_rf <- predict(model_rf, test)

# confusion matrix
cm_rf <- confusionMatrix(preds_rf, test$class)

# the importance of each variable is also accessible via the importance function
# polyuria is a clear winner, meaning it is likely to be the root note in most of the decision trees in the forest
importance(model_rf$finalModel)

# save rf accuracy and sensitivity
acc_rf <- cm_rf$overall["Accuracy"]
sen_rf <- cm_rf$byClass["Sensitivity"]

# store rf results
results <- bind_rows(results, tibble(Method = "Model (4) Random Forest", Accuracy = acc_rf , Sensitivity = sen_rf))
results %>% knitr::kable()




######____________Model 5 Ensemble _________

# this section creates an ensemble using the three best performing models
# since the decision tree performed the worst, it is dropped.
# it makes sense to drop the decision tree model since the random forest model is essentially an improved version

# store the predictions from the three models in a data frame

all_preds <- tibble(glm = preds_glm,
                    rf = preds_rf,
                    knn = preds_knn)

# the prediction of the ensemble are defined by the majority
preds_en <- apply(all_preds, 1, function(x) names(which.max(table(x)))) %>%
  factor(levels = c("Positive", "Negative"))

# confusion matrix (it actually performs worse than the random forest)
cm_ens <- confusionMatrix(preds_en, test$class)

# save accuracy and sensitivity
acc_ens <- cm_ens$overall["Accuracy"]
sen_ens <- cm_ens$byClass["Sensitivity"]

# store ensemble results
results <- bind_rows(results, tibble(Method = "Model (5) Ensemble", Accuracy = acc_ens , Sensitivity = sen_ens))
results %>% knitr::kable()



#############################
# final model - random forest
#############################

# the random forest had the highest accuracy and sensitivity out of all of the models, including the ensemble
# therefore, it is chosen to be the final model and is reconstructed using the diabetes data set

set.seed(1)
final_model_rf <- train(class~.,
                        method = "rf",
                        tuneGrid = data.frame(mtry = 3:11),
                        data = diabetes)

#visualize the performance of each mtry
ggplot(final_model_rf, highlight = TRUE)+
  scale_x_discrete(limits = 2:12) +
  ggtitle("Accuracy for each number of randomly selected predictors")

ggsave("rmd_files/images/mtry_final.png", width = 8, height = 5)
  
#best mtry is 5
final_model_rf$bestTune

#generating predictions
final_preds_rf <- predict(final_model_rf, validation)

cm_final <- confusionMatrix(final_preds_rf, validation$class)

#importance of each feature
imp_final <- importance(final_model_rf$finalModel)

# save accuracy and sensitivity
acc_final <- cm_final$overall["Accuracy"]
sen_final <- cm_final$byClass["Sensitivity"]

# store validation results
results <- bind_rows(results, tibble(Method = "Final Validation", Accuracy = acc_final , Sensitivity = sen_final))
results %>% knitr::kable()

