setwd('C:/Users/iceca/Documents/Earthquake_Damage_Predictor')
library(tidyverse)
library(nnet)
library(MASS)
library(caret)

#loadFin<- modules::use('Helpers/Load_Final_Data.R')
#train <- loadFin$original()[[1]]
#trainLab <- loadFin$original()[[2]]
#val <- loadFin$original()[[3]]
#valLab <- loadFin$original()[[4]]
#test <- loadFin$original()[[5]]

#trainLab$damage_grade <- as.factor(trainLab$damage_grade)
#valLab$damage_grade <- as.factor(valLab$damage_grade)


#manipulate<- modules::use('Helpers/Manipulate.R')
#trainFull <- manipulate$combineLab(train, trainLab)
#valFull <- manipulate$combineLab(val, valLab)

#trainFull$building_id <- NULL
#valFull$building_id <- NULL

#Neuralnet preprocessing

#one hot encode the label
trainLab <- data.frame(predict(dummyVars(" ~ .", data = trainLab), trainLab) )
valLab <- data.frame(predict(dummyVars(" ~ .", data = valLab), valLab))

#combine test, val, and train to preprocess together
ntrain <- nrow(train)
nval <- nrow(val)
ntest <- nrow(test)
X <- rbind(train, val, test)
building_id <- X$building_id #extract building_id because it should not be scaled (to assisst in joins)
X <- model.matrix(~.-building_id , X)
X <- scale(X)
#X <- cbind(X,building_id)
#trainProcessed <- inner_join(data.frame(X[1:ntrain,]), trainLab)
#valProcessed <- inner_join(data.frame(X[(1:nval) + ntrain ,]), valLab)
#testProcessed <- data.frame(X[(1:ntest) + ntrain + nval,])
X <- data.frame(cbind(X,building_id))
X$X.Intercept. <- NULL
trainProcessed <- inner_join(X[1:ntrain,], trainLab)
valProcessed <- inner_join(X[(1:nval) + ntrain ,], valLab)
testProcessed <- X[(1:ntest) + ntrain + nval,]


f = as.formula("damage_grade.1 + damage_grade.2 + damage_grade.3 ~. -building_id")
nn <- neuralnet(f,
                data = trainProcessed,
                hidden = c(13, 10, 3),threshold = 0.005,
                act.fct = "tanh",
                linear.output = FALSE,
                lifesign = "minimal")

#model.NeuralNet <- neuralnet(as.factor(damage_grade)~., data = trainFull,
#                             hidden=4,threshold=0.005, learningrate=0.005, )
#predictionsVal <- predict(model.NeuralNet, valFull)
#accuracy <- mean(predictionsVal$class == valFull$damage_grade)