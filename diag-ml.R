# adi
# 12/4/2016
# packages for DIAG ML session
set.seed(42)
setwd("~/MBA/DIAG/ml-workshop")
rm(list=ls())

#install.packages('ggplot2')
#install.packages('rpart')
#install.packages('randomForest')
#install.packages('caTools')
#install.packages('rpart.plot')
#install.packages('dplyr')
#install.packages('ROCR')

library(caTools)
library(dplyr)
library(randomForest)
library(rpart)
library(rpart.plot)
library(dplyr)
library(ROCR)
library(e1071)

data <- read.csv('data/labelled_data.csv')
unlabelled <- read.csv('data/unlabelled_data.csv')

data <- select(data,c(Survived,Pclass,Sex,Age,Fare))
data$Pclass <- as.factor(data$Pclass)
data$Survived <- as.factor(data$Survived)

spl <- sample.split(Y = data,SplitRatio = 3/4)

raw_train <- data[spl,]
raw_test <- data[!spl,]

comp <- complete.cases(raw_train)
train <- raw_train[comp,]

comp <- complete.cases(raw_test)
test <- raw_test[comp,]

# Logistic Regression
lm_model <- glm(Survived ~ .,data=train,family = 'binomial')
p <- predict(lm_model, newdata=test, type="response")
pr <- prediction(p, test$Survived)

# Create a AUC curve
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

# AUC for the model
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
print("#Logistic Regression")
print(auc)

# Decision Tree Model
rpart_model <- rpart(Survived ~ .,data=train, method='class')
rpart.plot(rpart_model)
rpart_pred <- predict(rpart_model,newdata = test,type = 'class')
print("#Classification and Regression Trees (CART)")
print(table(test$Survived,rpart_pred))

# Random Forest Model
rf_model <- randomForest(Survived ~.,data = train, method='class',ntree = 500, cutoff=c(.5,.5))
rf_pred <- predict(rf_model,newdata = test,type = 'class')
print("#Random Forest")
print(table(test$Survived,rf_pred))

# Support Vector Machines
svm_model <- svm(Survived ~ .,data=train)
svm_pred <- predict(svm_model,test)
print("#Support Vector Machines")
print(table(test$Survived,svm_pred))
