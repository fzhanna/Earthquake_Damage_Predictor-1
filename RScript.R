#IMPORTANT NOTE
#This script will not run. The reason is because development was done using modules.
#The code here is for the purposes of display only. Furthermore the if you read the
#notebook instead of script you will also have access to the markdown cells, which were used instead of code comments. This will help readibility.
#To run to code or for greater understanding/readibility, go to 
#https://github.com/icecap360/Earthquake_Damage_Predictor.

#The raw data was stored in the folder Earthquake_Damage_Predictor/Raw_Data/

#FILE NAME AND LOCATION: Earthquake_Damage_Predictor/Requirements/Requirements.ipynb

setwd('C:/Users/iceca/Documents/Earthquake_Damage_Predictor')
# Requirements for the development
install.packages("modules")
# Requirements for analysis
install.packages("corrplot")
install.packages("boot")
install.packages("gridExtra")
install.packages("MASS")
install.packages("speedglm")
install.packages("readr")
install.packages("ANN2")
install.packages("randomForest")
install.packages("e1071")
install.packages("parallelSVM")

#FILE NAME AND LOCATION: Earthquake_Damage_Predictor/Helpers/Load_Final_Data.R
getData <- function(prefix) {
  return(list(utils::read.csv(paste('Further_Preprocess_Analysis/Data/',prefix,'_train.csv', sep=''), stringsAsFactors = TRUE)[,-1], 
              utils::read.csv(paste('Further_Preprocess_Analysis/Data/',prefix,'_train_lab.csv', sep=''), stringsAsFactors = TRUE)[,-1],
              utils::read.csv(paste('Further_Preprocess_Analysis/Data/',prefix,'_val.csv', sep=''), stringsAsFactors = TRUE)[,-1],
              utils::read.csv(paste('Further_Preprocess_Analysis/Data/',prefix,'_val_lab.csv', sep=''), stringsAsFactors = TRUE)[,-1], 
              utils::read.csv(paste('Further_Preprocess_Analysis/Data/',prefix,'_test.csv', sep=''), stringsAsFactors = TRUE)[,-1]))
}
original <- function() {
  return(getData('original'))
}
expanded <- function() {
  return(getData('expanded'))
}
filtered <- function() {
  return(getData('expanded'))
}
a<-original()

#FILE NAME AND LOCATION: Earthquake_Damage_Predictor/Helpers/Load_Num_Factor.R

num <- function() {
  return(list(utils::read.csv('Preprocess_Analysis/Data/train_values_numeric.csv'), 
              utils::read.csv('Preprocess_Analysis/Data/train_labels_splitted.csv')))
}
factor <- function() {
  return(list(utils::read.csv('Preprocess_Analysis/Data/train_values_factor.csv'),
              utils::read.csv('Preprocess_Analysis/Data/train_labels_splitted.csv')))
}

#FILE NAME AND LOCATION: Earthquake_Damage_Predictor/Helpers/Load_Preprocessed.R

loadTrain <- function() {
  return(list(utils::read.csv('Preprocess_Analysis/Data/train_values_splitted.csv'),
              utils::read.csv('Preprocess_Analysis/Data/train_labels_splitted.csv')))
}
loadVal <- function() {
  return(list(utils::read.csv('Preprocess_Analysis/Data/val_values_splitted.csv'),
              utils::read.csv('Preprocess_Analysis/Data/val_labels_splitted.csv')))
}

#FILE NAME AND LOCATION: Earthquake_Damage_Predictor/Helpers/Load_Raw_Data.R

trainVal <- function() {
  return(utils::read.csv('Raw_Data/train_values.csv'))
}
testVal <- function() {
  return(utils::read.csv('Raw_Data/test_values.csv'))
}
trainLab <- function() {
  return(utils::read.csv('Raw_Data/train_labels.csv'))
}

#FILE NAME AND LOCATION: Earthquake_Damage_Predictor/Helpers/Manipulate.R

combineLab <- function(data, label) {
  return(dplyr::inner_join(data, label))
}

#FILE NAME AND LOCATION: Earthquake_Damage_Predictor/Helpers/Split.R

trainVal <- function(seed, prob, fullValues, fullLabels) {
  valIndex <- caret::createDataPartition(fullLabels[,1], p=prob, 
                                         list = FALSE, times = 1)
  return(list(fullValues[-valIndex,], fullLabels[-valIndex,], 
              fullValues[valIndex,], fullLabels[valIndex,]))
}

#FILE NAME AND LOCATION: Earthquake_Damage_Predictor/Preprocess_Analysis/Preprocess.ipnyb

library(tidyverse)
setwd('C:/Users/iceca/Documents/Earthquake_Damage_Predictor/')
#Load Data
loadRaw <- modules::use("Helpers/Load_Raw_Data.R")
train <- loadRaw$trainVal()
test <- loadRaw$testVal()
lab <- loadRaw$trainLab()
train_sub <- train[50:100,]
split <- modules::use("Helpers/Split.R")
valPercentage = 0.1
seed = 0

head(test)
for (i  in 1:length(train[1,])) {
  print(names(train)[i])
  print(train[1,i])
}
cat('ncol test', ncol(test), '\n')
cat('ncol train', ncol(train),'\n')
cat('ncol labels', ncol(lab), '\n')
cat('nrow train', nrow(train), '\n')
cat('nrow test', nrow(test), '\n')
cat('nrow labels', nrow(lab), '\n')

cat('train is same as tests:', all( names(train) == names(test)), '\n')

cat('Feature types \n')
for (i  in 1:length(train)) {
  cat('feature',names(train)[i],'type is   ',class(train[,i]), '\n')
}

#Missing data, there is none
apply(head(is.na(train)),2, sum)

#Making a training validation split
splittedData <- split$trainVal(seed,valPercentage, train, lab)
train_splitted <- splittedData[[1]]
write.csv(splittedData[[1]], 'Preprocess_Analysis/Data/train_values_splitted.csv')
write.csv(splittedData[[2]], 'Preprocess_Analysis/Data/train_labels_splitted.csv')
write.csv(splittedData[[3]], 'Preprocess_Analysis/Data/val_values_splitted.csv')
write.csv(splittedData[[4]], 'Preprocess_Analysis/Data/val_labels_splitted.csv')

dim(splittedData[[1]])
dim(splittedData[[2]])
dim(splittedData[[3]])
dim(splittedData[[4]])

#Factor feature statistics
factorCols <- names(train_splitted)[unlist(lapply(train_splitted, is.factor))]
trainFactors <- train_splitted %>% select(factorCols)
head(trainFactors)
write.csv(trainFactors, 'Preprocess_Analysis/Data/train_values_factor.csv')
#plan_configuration contains too many levels, 10
#foundation_type, ground_floor_type have 5

#Numeric Data
numCols <- names(train_splitted)[unlist(lapply(train_splitted, is.numeric))]
trainNum <- train_splitted %>% select(numCols)
head(trainNum)
write.csv(trainNum, 'Preprocess_Analysis/Data/train_values_numeric.csv')

#FILE NAME AND LOCATION: Earthquake_Damage_Predictor/Preprocess_Analysis/Analysis.ipnyb
setwd('C:/Users/iceca/Documents/Earthquake_Damage_Predictor/')
library(tidyverse)
library(ggplot2)
library(boot)
library(gridExtra)
library(corrplot)

loadNF <- modules::use('Helpers/Load_Num_Factor.R')
manipulate <- modules::use('Helpers/Manipulate.R')
numTrain <- loadNF$num()[[1]]
factorTrain <- loadNF$factor()[[1]]
labels <- loadNF$factor()[[2]] #num and factor features have the same labels
numTrain <- manipulate$combineLab(numTrain, labels)
factorTrain <- manipulate$combineLab(factorTrain, labels)



corNumTrain <- cor(numTrain)
cat('\n Features that show greatest corrrlation with each other \n \n')
corrplot(corNumTrain)
for (i in 1:length(names(numTrain))) {
  for (j in 1:length(names(numTrain))) {
    
    if (abs(corNumTrain[i,j]) >0.3 && abs(corNumTrain[i,j]) <1.0 && i>=j) {
      cat(names(numTrain)[i] , names(numTrain)[j], corNumTrain[i,j], '\n')
    }
  }
}
labColNumber = 33
thr = 0.1
releventFeat <- sort( corNumTrain[,labColNumber][abs(corNumTrain[,labColNumber])>thr])
cat('\n Features that show greatest corrrlation with the label ')
names(releventFeat)
cat(releventFeat)

print( table(numTrain$has_superstructure_timber , numTrain$has_superstructure_bamboo)/(
  length(numTrain$has_superstructure_timber)+length(numTrain$has_superstructure_bamboo)) )

cat('Note that the odd shape is because floors is discrete')
numTrain %>% ggplot(aes(count_floors_pre_eq)) +
  geom_density(fill = 'red',alpha=0.8,adjust=4) +
  ggtitle('Floors is right skewed')

numTrain %>% ggplot(aes(area_percentage)) + geom_density(fill = 'red',alpha=0.8, adjust=2) +  ggtitle('Area follows a normal distribution')

numTrain %>% ggplot(aes(age)) + geom_density(fill = 'red',alpha=0.8, adjust=3) +  ggtitle('Age follows a normal distribution')

numTrain %>% ggplot(aes(height_percentage)) + geom_density(fill = 'red',alpha=0.8, adjust=2) +  ggtitle('Height percentage follows a normal distribution')

numTrain %>% ggplot(aes(count_families)) + geom_density(fill = 'red',alpha=0.8, adjust=2) +  ggtitle('Family follows a normal distribution')

binaryFeat <- apply(numTrain,2,function(x) { all(x %in% 0:1) })
binTrain <- numTrain[,binaryFeat]
head(binTrain)

cat('How many distinct buildings have these features?')
apply(binTrain, 2, sum)#/nrow(binTrain)

#Investigating the interaction between binary features
x <- matrix(1:9, nrow = 3, dimnames = list(c("X","Y","Z"), c("A","B","C")))
nbin <- ncol(binTrain)
nameBin <- names(binTrain)
propInteraction <- matrix(1:nbin^2, nrow=nbin, dimnames=list(nameBin,nameBin))

for (i in 1:length(names(binTrain))) {
  for (j in 1:length(names(binTrain))) {
    freq <- prop.table(table(binTrain[,i], binTrain[,j]))
    propInteraction[i,j] = freq[2,2]/(freq[1,2]+freq[2,1])
    #interactionPval[i,j] =  summary(freq)$p.value
  }
}

hist(as.vector(propInteraction), breaks=100)

#Investigating the relationship between binary variables and the label
x <- matrix(1:9, nrow = 3, dimnames = list(c("X","Y","Z"), c("A","B","C")))
interactionPval <- 1:nbin

names(interactionPval) <- nameBin
for (i in 1:length(names(binTrain))) {
  freq <- table(binTrain[,i], numTrain$damage_grade)
  cat('Table for variable ', nameBin[i])
  print(prop.table(freq))    
  cat('\n')
  interactionPval[i] =  summary(freq)$p.value
}
interactionPval

binTrain %>% 
  mutate(damage_grade =labels$damage_grade) ->
  binTrain2
nameBin2 <- names(binTrain2)
numFeat <- length(nameBin2)
plotBinLab <- function(start,end, nameBin2, binTrain2) {
  n <- floor(sqrt(end-start+1))
  par(mfrow = c(n,n))
  for (i in start:end) {
    freqFeat <- prop.table(table(binTrain2[,nameBin2[i]], numTrain$damage_grade), 1)
    freqData <- prop.table(table(binTrain2$damage_grade))
    freq <- rbind(freqFeat[2,],freqData)
    barplot( freq , beside=TRUE, main = nameBin2[i] )
  }
}

cat('The purpose of these figures is to compare the distribution of damage_grade among those that have a
    particular feature and that of the whole data. Hence to determine if a feature is a good predictor, look for
    diffferences among the colors')
plotBinLab(1,8, nameBin2, binTrain2)
plotBinLab(9,16, nameBin2, binTrain2)
plotBinLab(17,22, nameBin2, binTrain2)

factorTrain$X <- NULL
factorTrain$building_id <- NULL
facs <- names(factorTrain)
facs <- setdiff(facs, c('damage_grade')) 
for (name in facs) {
  cat('Analyzing' , name, '\n')
  feature <- factorTrain[,name]
  print(table(feature))
  cat('\n Distribution of feature among buildings of the same damage grade (marginal percentage taken along column)')
  tab <-round(prop.table(table(feature,factorTrain$damage_grade),2)*100,0)
  print(tab)
  cat('\n Distribution of buliding with the same feature value (marginal percentage taken along rows)')
  tab <-round(prop.table(table(feature,factorTrain$damage_grade),1)*100,0)
  print(tab)
  cat('\n \n')
}

#FILE NAME AND LOCATION: Earthquake_Damage_Predictor/Further_Preprocess_Analysis/Preprocess.ipnyb

setwd('C:/Users/iceca/Documents/Earthquake_Damage_Predictor/')
library(tidyverse)

loadPr<- modules::use('Helpers/Load_Preprocessed.R')
train <- loadPr$loadTrain()[[1]]
trainLab <- loadPr$loadTrain()[[2]]

val <- loadPr$loadVal()[[1]]
valLab <- loadPr$loadVal()[[2]]

loadRaw<- modules::use('Helpers/Load_Raw_Data.R')
test <- loadRaw$testVal()

removeId <- function(data) {
  data$X <- NULL
  return(data)
}
removeLevelsPlan <- function(data) {
  newLevel = "other"
  data$plan_configuration <- plyr::revalue(data$plan_configuration, 
                                           c("a"=newLevel, "c"=newLevel, "f"=newLevel,
                                             "m"=newLevel, "n"=newLevel, "o"=newLevel, 
                                             "s" = newLevel))
  return(data)
}
labelToFactor <- function(data) {
  data$damage_grade <- as.factor(data$damage_grade)
  return(data)
}
saveData <- function(tr, trLab, val, valLab, test, prefix) {
  tr <- removeId(tr)
  trLab <- labelToFactor(removeId(trLab)) #labels do not have plan configuration to removeLevelsPlan not called
  val <- removeId(val)
  valLab <- labelToFactor(removeId(valLab)) #labels do not have plan configuration to removeLevelsPlan not called
  test <- removeId(test) #do not remove the building_ids of the test, needed for submission
  if ("plan_configuration" %in% names(tr)) { #training set was used here but any dataset could be used instead
    tr <- removeLevelsPlan(tr)
    val <- removeLevelsPlan(val)
    test <- removeLevelsPlan(test)
  }
  rootDir <- 'Further_Preprocess_Analysis/Data/'
  write.csv(tr, paste(rootDir, prefix, '_train.csv', sep=''))
  write.csv(trLab, paste(rootDir, prefix, '_train_lab.csv', sep=''))
  write.csv(val, paste(rootDir, prefix, '_val.csv', sep=''))
  write.csv(valLab, paste(rootDir, prefix, '_val_lab.csv', sep=''))
  write.csv(test, paste(rootDir, prefix, '_test.csv', sep=''))
}

saveData(train, trainLab, val, valLab, test, 'original')

featureEngineerAdd <- function(data) {
  data$has_superstructure_tree = 
    data$has_superstructure_bamboo | data$has_superstructure_timber
  data$has_superstructure_mortar = 
    data$has_superstructure_mud_mortar_stone | data$has_superstructure_cement_mortar_stone | data$has_superstructure_mud_mortar_brick | data$has_superstructure_cement_mortar_brick
  data$has_superstructure_cement = 
    data$has_superstructure_cement_mortar_stone | data$has_superstructure_timber
  data$has_superstructure_brick = 
    data$has_superstructure_mud_mortar_brick | data$has_superstructure_cement_mortar_brick
  data$has_superstructure_mud = 
    data$has_superstructure_adobe_mud | data$has_superstructure_mud_mortar_stone | data$has_superstructure_mud_mortar_brick
  data$has_superstructure_concrete = 
    data$has_superstructure_rc_non_engineered | data$has_superstructure_rc_engineered 
  data$has_superstructure_stone = 
    data$has_superstructure_mud_mortar_stone | data$has_superstructure_stone_flag | data$has_superstructure_cement_mortar_stone
  return(data)
}

saveData(featureEngineerAdd(train), trainLab, 
         featureEngineerAdd(val), valLab, featureEngineerAdd(test), 'expanded')

featureEngineerRemove <- function(data) {
  newLevel <- "other"
  #according to analysis, combine levels that were strikingly similar
  data$foundation_type <- plyr::revalue(data$foundation_type, c("u"=newLevel, "w"=newLevel))
  data$roof_type <- plyr::revalue(data$roof_type, c("n"=newLevel, "q"=newLevel))    
  data %>% select(
    #first add all binary/categorical features that were found to be effective in the analysis stage
    foundation_type , roof_type , ground_floor_type, other_floor_type, legal_ownership_status, 
    has_superstructure_stone_flag, has_superstructure_cement_mortar_stone, 
    has_superstructure_cement_mortar_brick, has_superstructure_mud_mortar_stone, has_superstructure_rc_non_engineered, 
    has_superstructure_rc_engineered, has_superstructure_timber,has_secondary_use, has_secondary_use_hotel, has_secondary_use_rental, 
    has_secondary_use_institution, has_secondary_use_industry, 
    plan_configuration,
    #now add all other features that were considered relevent according to research conducted on the internet
    geo_level_1_id, geo_level_2_id, geo_level_3_id, count_floors_pre_eq, age, 
    area_percentage, height_percentage, land_surface_condition, count_families, building_id) %>% return()
}

saveData(featureEngineerRemove(train), trainLab, 
         featureEngineerRemove(val), valLab, featureEngineerRemove(test), 'filtered')

#FILE NAME AND LOCATION: Earthquake_Damage_Predictor/Models/Original_Data_Models.ipnyb

setwd('C:/Users/iceca/Documents/Earthquake_Damage_Predictor')
library(tidyverse)
library(MASS)
library(caret)
library(randomForest)
library(e1071)

loadFin<- modules::use('Helpers/Load_Final_Data.R')
train <- loadFin$original()[[1]]
trainLab <- loadFin$original()[[2]]
val <- loadFin$original()[[3]]
valLab <- loadFin$original()[[4]]
test <- loadFin$original()[[5]]

#when R reads the csv, it thinks damage_grade is an integer, so convert it back to a factor
trainLab$damage_grade <- as.factor(trainLab$damage_grade)
valLab$damage_grade <- as.factor(valLab$damage_grade)

manipulate<- modules::use('Helpers/Manipulate.R')
trainFull <- manipulate$combineLab(train, trainLab)
valFull <- manipulate$combineLab(val, valLab)

#saving predictions helper function
savePredictions <- function(model) {
  preds <- cbind(test$building_id, predict(model, subset(test, select=-c(building_id))))
  colnames(preds) <- c("building_id", "damage_grade")
  write.csv(preds, 'Models/Predictions/Random_Forest.csv', row.names=FALSE)
}

trainFull$building_id <- NULL
valFull$building_id <- NULL

model.Multinom <- multinom(as.factor(damage_grade) ~ ., trainFull, maxit=300)

prediction <- predict(model.Multinom, valFull)
accuracy <- mean(prediction == valFull$damage_grade)
cat('Accuracy for the neural network was ', accuracy)

model.LDA <- lda(as.factor(damage_grade)~., data = trainFull)
predictionsVal <- predict(model.LDA, valFull)
accuracy <- mean(predictionsVal$class == valFull$damage_grade)
cat('Accuracy for the LDA was ', accuracy)

model.QDA <- qda(as.factor(damage_grade)~., 
                 data = trainFull)
predictionsVal <- predict(model.QDA, valFull)
accuracy <- mean(predictionsVal$class == valFull$damage_grade)

#Neuralnet preprocessing

#one hot encode the label
trainLabProcessed <- data.frame(predict(dummyVars(" ~ .", data = trainLab), trainLab) )
valLabProcessed <- data.frame(predict(dummyVars(" ~ .", data = valLab), valLab))

#combine test, val, and train to preprocess together
ntrain <- nrow(train)
nval <- nrow(val)
ntest <- nrow(test)
X <- rbind(train, val, test)
building_id <- X$building_id #extract building_id because it should not be scaled (to assisst in joins)
X <- model.matrix(~.-building_id , X)
X <- scale(X)
X <- data.frame(cbind(X,building_id))
X$X.Intercept. <- NULL
#trainProcessed <- subset( inner_join(X[1:ntrain,], trainLabProcessed) , select=-c(building_id)) 
#valProcessed <- subset(inner_join(X[(1:nval) + ntrain ,], valLabProcessed), select=-c(building_id))
#testProcessed <- subset(X[(1:ntest) + ntrain + nval,], select=-c(building_id))
trainProcessed <- subset(X[1:ntrain,], select=-c(building_id))
valProcessed <- subset(X[(1:nval) + ntrain ,], select=-c(building_id))
testProcessed <- subset(X[(1:ntest) + ntrain + nval,], select=-c(building_id))
trainLab <- subset(trainLab, select=-c(building_id))
valLab <- subset(valLab, select=-c(building_id))

model.Neuralnet <- ANN2::neuralnetwork(trainProcessed, trainLab, lossFunction = "log",
                                       rectifierLayers = NA, sigmoidLayers = NA, regression = FALSE,
                                       standardize = TRUE, learnRate = 5e-03, maxEpochs = 10,
                                       hiddenLayers = c(5, 5), momentum = 0.9, learnRate = 0.001, verbose = TRUE)

#f = as.formula("damage_grade.1 + damage_grade.2 + damage_grade.3 ~. -building_id")
#nn <- neuralnet(f,
#                data = trainProcessed,
#                hidden = c(5, 5, 5), threshold = 0.001,
#                act.fct = "tanh",
#                linear.output = FALSE,
#                lifesign = "minimal")

mtry <- c(5,10,15,20,25)
ntree <- c(80,70,60,50,40) 

createRandomForest <- function(mtry_, ntree_, trainFull, valFull){
  model.RF <- randomForest(as.factor(damage_grade) ~ ., data=trainFull, ntree=ntree_, mtry=mtry_, importance=TRUE)
  #dput(list(model.RF$confusion, modelRF$oob.times, modelRF$, "Models/Random_Forest_Output.txt")
  cat('RESULTS FOR RANDOM FOREST, with mtry=', mtry_, 'ntree=', ntree_, '\n \n')
  
  predictionsVal <- predict(model.RF, valFull)
  accuracy <- mean(predictionsVal == valFull$damage_grade)
  cat('The accuracy for this random forest was',accuracy, '\n')
  
  cat('\n Confusion Matrix \n')
  print(model.RF$confusion)
  
  cat('\n Variables sorted in importance (decreasing)\n')
  imp <- as.data.frame(importance(model.RF))
  print(subset ( imp[order(-imp$MeanDecreaseGini),] ,select=MeanDecreaseGini))
}

createRandomForest(mtry[1],ntree[1], trainFull, valFull)

createRandomForest(mtry[2],ntree[2], trainFull, valFull)

createRandomForest(mtry[3],ntree[3], trainFull, valFull)

createRandomForest(mtry[4],ntree[4], trainFull, valFull)

createRandomForest(mtry[5],ntree[5], trainFull, valFull)

#best model based on accuracy of the previous models
model.RF <- randomForest(as.factor(damage_grade) ~ ., data=rbind(trainFull,valFull), ntree=ntree[1], mtry=mtry[1], importance=FALSE)


savePredictions(model.RF)

createSVMModel <- function(kernel, cost, trainFull, valFull, gamma=0.1, degree=1) {
  if (kernel == "linear"){
    cat("Kernel selected as linear")
    model.SVM <- svm(as.factor(damage_grade) ~., data=trainFull, cost=cost, scale=TRUE, kernel="linear")
  }
  if (kernel == "polynomial") {
    cat("Kernel selected as polynomial")
    model.SVM  <- svm(as.factor(damage_grade) ~., data=trainFull, cost=cost, degree=degree, scale=TRUE, kernel="polynomial")
  }
  if (kernel == "radial") {
    cat("Kernel selected as radial")
    model.SVM <- svm(as.factor(damage_grade) ~., data=trainFull, cost=cost,gamma=gamma, scale=TRUE, kernel="radial")
  }
  preds <- predict(model.SVM, valFull)
  acc <- mean(preds == valFull$damage_grade)
  cat('The accuracy for this SVM, with kernel', kernel, 'and cost', cost,'\n\n')
}
gammaSearch <- 10^(-9:3)
costSearch <- 10^(-3:3)
degreeSearch <- 1:5

library(parallelSVM)
model.SVM <- parallelSVM(as.factor(damage_grade) ~., data=trainFull, sampleSize=0.1,kernel="linear", cost=0.5)

preds <- predict(model.SVM, valFull)
acc <- mean(preds == valFull$damage_grade)

cat('The accuracy for this SVM, with kernel','linear', 'and cost', 0.5,'was',acc,'\n\n')

#linear search
for (c in c(0.01, 1, 10)){ 
  createSVMModel("linear", c, trainFull, valFull) 
}

#polynomial
polyParams <- expand.grid(costSearch, degreeSearch)
ntry <- 10
set.seed(5)
params <- sample(1:nrow(polyParams) , ntry, replace=FALSE)
for (i in params){
  c <- polyParams[i,]$cost
  deg <- polyParams[i,]$degree
  cat('PARAMETERS: COST',c,' DEGREE ',deg, '\n')
  createSVMModel("polynomial", c, trainFull, valFull, degree=deg) 
}

#radial
radParams <- expand.grid(cost=costSearch, gamme=gammaSearch)
ntry <- 10
set.seed(5)
params <- sample(1:nrow(radParams) , ntry, replace=FALSE)
for (i in params){
  c <- radParams[i,]$cost
  gam <- radParams[i,]$gamma
  createSVMModel("polynomial", c, trainFull, valFull, gamma=gam) 
  cat('PARAMETERS: COST',c,' GAMMA ',gam, '\n')
}

#FILE NAME AND LOCATION: Earthquake_Damage_Predictor/Models/Filtered_Data_Models.ipnyb

setwd('C:/Users/iceca/Documents/Earthquake_Damage_Predictor')
library(tidyverse)
library(MASS)
library(caret)
library(nnet)
library(randomForest)
library(e1071)

loadFin<- modules::use('Helpers/Load_Final_Data.R')
train <- loadFin$filtered()[[1]]
trainLab <- loadFin$filtered()[[2]]
val <- loadFin$filtered()[[3]]
valLab <- loadFin$filtered()[[4]]
test <- loadFin$filtered()[[5]]

#when R reads the csv, it thinks damage_grade is an integer, so convert it back to a factor
trainLab$damage_grade <- as.factor(trainLab$damage_grade)
valLab$damage_grade <- as.factor(valLab$damage_grade)

manipulate<- modules::use('Helpers/Manipulate.R')
trainFull <- manipulate$combineLab(train, trainLab)
valFull <- manipulate$combineLab(val, valLab)

#saving predictions helper function
savePredictions <- function(model) {
  preds <- cbind(test$building_id, predict(model, subset(test, select=-c(building_id))))
  colnames(preds) <- c("building_id", "damage_grade")
  write.csv(preds, 'Models/Predictions/Random_Forest.csv', row.names=FALSE)
}

trainFull$building_id <- NULL
valFull$building_id <- NULL

model.Multinom <- multinom(as.factor(damage_grade) ~ ., trainFull, maxit=300)

prediction <- predict(model.Multinom, valFull)
accuracy <- mean(prediction == valFull$damage_grade)
cat('Accuracy for the neural network was ', accuracy)

model.LDA <- lda(as.factor(damage_grade)~., data = trainFull)
predictionsVal <- predict(model.LDA, valFull)
accuracy <- mean(predictionsVal$class == valFull$damage_grade)
cat('Accuracy for the LDA was ', accuracy)

model.QDA <- qda(as.factor(damage_grade)~., 
                 data = trainFull)
predictionsVal <- predict(model.QDA, valFull)
accuracy <- mean(predictionsVal$class == valFull$damage_grade)
accuracy

#Neuralnet preprocessing

#one hot encode the label
trainLabProcessed <- data.frame(predict(dummyVars(" ~ .", data = trainLab), trainLab) )
valLabProcessed <- data.frame(predict(dummyVars(" ~ .", data = valLab), valLab))

#combine test, val, and train to preprocess together
ntrain <- nrow(train)
nval <- nrow(val)
ntest <- nrow(test)
X <- rbind(train, val, test)
building_id <- X$building_id #extract building_id because it should not be scaled (to assisst in joins)
X <- model.matrix(~.-building_id , X)
X <- scale(X)
X <- data.frame(cbind(X,building_id))
X$X.Intercept. <- NULL
#trainProcessed <- subset( inner_join(X[1:ntrain,], trainLabProcessed) , select=-c(building_id)) 
#valProcessed <- subset(inner_join(X[(1:nval) + ntrain ,], valLabProcessed), select=-c(building_id))
#testProcessed <- subset(X[(1:ntest) + ntrain + nval,], select=-c(building_id))
trainProcessed <- subset(X[1:ntrain,], select=-c(building_id))
valProcessed <- subset(X[(1:nval) + ntrain ,], select=-c(building_id))
testProcessed <- subset(X[(1:ntest) + ntrain + nval,], select=-c(building_id))
trainLab <- subset(trainLab, select=-c(building_id))
valLab <- subset(valLab, select=-c(building_id))

model.Neuralnet <- ANN2::neuralnetwork(trainProcessed, trainLab, lossFunction = "log",
                                       rectifierLayers = NA, sigmoidLayers = NA, regression = FALSE,
                                       standardize = TRUE, learnRate = 5e-03, maxEpochs = 10,
                                       hiddenLayers = c(5, 5), momentum = 0.9, learnRate = 0.001, verbose = TRUE)

#f = as.formula("damage_grade.1 + damage_grade.2 + damage_grade.3 ~. -building_id")
#nn <- neuralnet(f,
#                data = trainProcessed,
#                hidden = c(5, 5, 5), threshold = 0.001,
#                act.fct = "tanh",
#                linear.output = FALSE,
#                lifesign = "minimal")

mtry <- c(5,10,15,20,25)
ntree <- c(80,70,60,50,40) 

createRandomForest <- function(mtry_, ntree_, trainFull, valFull){
  model.RF <- randomForest(as.factor(damage_grade) ~ ., data=trainFull, ntree=ntree_, mtry=mtry_, importance=TRUE)
  #dput(list(model.RF$confusion, modelRF$oob.times, modelRF$, "Models/Random_Forest_Output.txt")
  cat('RESULTS FOR RANDOM FOREST, with mtry=', mtry_, 'ntree=', ntree_, '\n \n')
  
  predictionsVal <- predict(model.RF, valFull)
  accuracy <- mean(predictionsVal == valFull$damage_grade)
  cat('The accuracy for this random forest was',accuracy, '\n')
  
  cat('\n Confusion Matrix \n')
  print(model.RF$confusion)
  
  cat('\n Variables sorted in importance (decreasing)\n')
  imp <- as.data.frame(importance(model.RF))
  print(subset ( imp[order(-imp$MeanDecreaseGini),] ,select=MeanDecreaseGini))
}

createRandomForest(mtry[1],ntree[1], trainFull, valFull)

createRandomForest(mtry[2],ntree[2], trainFull, valFull)

createRandomForest(mtry[3],ntree[3], trainFull, valFull)

createRandomForest(mtry[4],ntree[4], trainFull, valFull)

createRandomForest(mtry[5],ntree[5], trainFull, valFull)

#best model based on accuracy of the previous models
model.RF <- randomForest(as.factor(damage_grade) ~ ., data=rbind(trainFull,valFull), ntree=ntree[2], mtry=mtry[2], importance=FALSE)

savePredictions(model.RF)

createSVMModel <- function(kernel, cost, trainFull, valFull, gamma=0.1, degree=1) {
  if (kernel == "linear"){
    cat("Kernel selected as linear")
    model.SVM <- svm(as.factor(damage_grade) ~., data=trainFull, cost=cost, scale=TRUE, kernel="linear")
  }
  if (kernel == "polynomial") {
    cat("Kernel selected as polynomial")
    model.SVM  <- svm(as.factor(damage_grade) ~., data=trainFull, cost=cost, degree=degree, scale=TRUE, kernel="polynomial")
  }
  if (kernel == "radial") {
    cat("Kernel selected as radial")
    model.SVM <- svm(as.factor(damage_grade) ~., data=trainFull, cost=cost,gamma=gamma, scale=TRUE, kernel="radial")
  }
  preds <- predict(model.SVM, valFull)
  acc <- mean(preds == valFull$damage_grade)
  cat('The accuracy for this SVM, with kernel', kernel, 'and cost', cost,'\n\n')
}
gammaSearch <- 10^(-9:3)
costSearch <- 10^(-3:3)
degreeSearch <- 1:5

library(parallelSVM)
model.SVM <- parallelSVM(as.factor(damage_grade) ~., data=trainFull, sampleSize=0.1,kernel="linear", cost=1)

preds <- predict(model.SVM, valFull)
acc <- mean(preds == valFull$damage_grade)


cat('The accuracy for this SVM, with kernel','linear', 'and cost', 0.5,'was',acc,'\n\n')

#linear search
for (c in c(0.01, 1, 10)){ 
  createSVMModel("linear", c, trainFull, valFull) 
}

#polynomial
polyParams <- expand.grid(costSearch, degreeSearch)
ntry <- 10
set.seed(5)
params <- sample(1:nrow(polyParams) , ntry, replace=FALSE)
for (i in params){
  c <- polyParams[i,]$cost
  deg <- polyParams[i,]$degree
  cat('PARAMETERS: COST',c,' DEGREE ',deg, '\n')
  createSVMModel("polynomial", c, trainFull, valFull, degree=deg) 
}

#radial
radParams <- expand.grid(cost=costSearch, gamme=gammaSearch)
ntry <- 10
set.seed(5)
params <- sample(1:nrow(radParams) , ntry, replace=FALSE)
for (i in params){
  c <- radParams[i,]$cost
  gam <- radParams[i,]$gamma
  createSVMModel("polynomial", c, trainFull, valFull, gamma=gam) 
  cat('PARAMETERS: COST',c,' GAMMA ',gam, '\n')
}

#FILE NAME AND LOCATION: Earthquake_Damage_Predictor/Models/Expanded_Data_Models.ipnyb

setwd('C:/Users/iceca/Documents/Earthquake_Damage_Predictor')
library(tidyverse)
library(MASS)
library(caret)
library(nnet)
library(randomForest)
library(e1071)

loadFin<- modules::use('Helpers/Load_Final_Data.R')
train <- loadFin$expanded()[[1]]
trainLab <- loadFin$expanded()[[2]]
val <- loadFin$expanded()[[3]]
valLab <- loadFin$expanded()[[4]]
test <- loadFin$expanded()[[5]]

#when R reads the csv, it thinks damage_grade is an integer, so convert it back to a factor
trainLab$damage_grade <- as.factor(trainLab$damage_grade)
valLab$damage_grade <- as.factor(valLab$damage_grade)

manipulate<- modules::use('Helpers/Manipulate.R')
trainFull <- manipulate$combineLab(train, trainLab)
valFull <- manipulate$combineLab(val, valLab)

#saving predictions helper function
savePredictions <- function(model) {
  preds <- cbind(test$building_id, predict(model, subset(test, select=-c(building_id))))
  colnames(preds) <- c("building_id", "damage_grade")
  write.csv(preds, 'Models/Predictions/Random_Forest.csv', row.names=FALSE)
}

trainFull$building_id <- NULL
valFull$building_id <- NULL

model.Multinom <- multinom(as.factor(damage_grade) ~ ., trainFull, maxit=300)

prediction <- predict(model.Multinom, valFull)
accuracy <- mean(prediction == valFull$damage_grade)
cat('Accuracy for the neural network was ', accuracy)

model.LDA <- lda(as.factor(damage_grade)~., data = trainFull)
predictionsVal <- predict(model.LDA, valFull)
accuracy <- mean(predictionsVal$class == valFull$damage_grade)
cat('Accuracy for the LDA was ', accuracy)

model.QDA <- qda(as.factor(damage_grade)~., 
                 data = trainFull)
predictionsVal <- predict(model.QDA, valFull)
accuracy <- mean(predictionsVal$class == valFull$damage_grade)
accuracy

#Neuralnet preprocessing

#one hot encode the label
trainLabProcessed <- data.frame(predict(dummyVars(" ~ .", data = trainLab), trainLab) )
valLabProcessed <- data.frame(predict(dummyVars(" ~ .", data = valLab), valLab))

#combine test, val, and train to preprocess together
ntrain <- nrow(train)
nval <- nrow(val)
ntest <- nrow(test)
X <- rbind(train, val, test)
building_id <- X$building_id #extract building_id because it should not be scaled (to assisst in joins)
X <- model.matrix(~.-building_id , X)
X <- scale(X)
X <- data.frame(cbind(X,building_id))
X$X.Intercept. <- NULL
#trainProcessed <- subset( inner_join(X[1:ntrain,], trainLabProcessed) , select=-c(building_id)) 
#valProcessed <- subset(inner_join(X[(1:nval) + ntrain ,], valLabProcessed), select=-c(building_id))
#testProcessed <- subset(X[(1:ntest) + ntrain + nval,], select=-c(building_id))
trainProcessed <- subset(X[1:ntrain,], select=-c(building_id))
valProcessed <- subset(X[(1:nval) + ntrain ,], select=-c(building_id))
testProcessed <- subset(X[(1:ntest) + ntrain + nval,], select=-c(building_id))
trainLab <- subset(trainLab, select=-c(building_id))
valLab <- subset(valLab, select=-c(building_id))

model.Neuralnet <- ANN2::neuralnetwork(trainProcessed, trainLab, lossFunction = "log",
                                       rectifierLayers = NA, sigmoidLayers = NA, regression = FALSE,
                                       standardize = TRUE, learnRate = 5e-03, maxEpochs = 10,
                                       hiddenLayers = c(5, 5), momentum = 0.9, learnRate = 0.001, verbose = TRUE)

#f = as.formula("damage_grade.1 + damage_grade.2 + damage_grade.3 ~. -building_id")
#nn <- neuralnet(f,
#                data = trainProcessed,
#                hidden = c(5, 5, 5), threshold = 0.001,
#                act.fct = "tanh",
#                linear.output = FALSE,
#                lifesign = "minimal")

mtry <- c(5,10,15,20,25)
ntree <- c(80,70,60,50,40) 

createRandomForest <- function(mtry_, ntree_, trainFull, valFull){
  model.RF <- randomForest(as.factor(damage_grade) ~ ., data=trainFull, ntree=ntree_, mtry=mtry_, importance=TRUE)
  #dput(list(model.RF$confusion, modelRF$oob.times, modelRF$, "Models/Random_Forest_Output.txt")
  cat('RESULTS FOR RANDOM FOREST, with mtry=', mtry_, 'ntree=', ntree_, '\n \n')
  
  predictionsVal <- predict(model.RF, valFull)
  accuracy <- mean(predictionsVal == valFull$damage_grade)
  cat('The accuracy for this random forest was',accuracy, '\n')
  
  cat('\n Confusion Matrix \n')
  print(model.RF$confusion)
  
  cat('\n Variables sorted in importance (decreasing)\n')
  imp <- as.data.frame(importance(model.RF))
  print(subset ( imp[order(-imp$MeanDecreaseGini),] ,select=MeanDecreaseGini))
}

createRandomForest(mtry[1],ntree[1], trainFull, valFull)

createRandomForest(mtry[2],ntree[2], trainFull, valFull)

createRandomForest(mtry[3],ntree[3], trainFull, valFull)

createRandomForest(mtry[4],ntree[4], trainFull, valFull)

createRandomForest(mtry[5],ntree[5], trainFull, valFull)

#best model based on accuracy of the previous models
model.RF <- randomForest(as.factor(damage_grade) ~ ., data=rbind(trainFull,valFull), ntree=ntree[2], mtry=mtry[2], importance=FALSE)

savePredictions(model.RF)

createSVMModel <- function(kernel, cost, trainFull, valFull, gamma=0.1, degree=1) {
  if (kernel == "linear"){
    cat("Kernel selected as linear")
    model.SVM <- svm(as.factor(damage_grade) ~., data=trainFull, cost=cost, scale=TRUE, kernel="linear")
  }
  if (kernel == "polynomial") {
    cat("Kernel selected as polynomial")
    model.SVM  <- svm(as.factor(damage_grade) ~., data=trainFull, cost=cost, degree=degree, scale=TRUE, kernel="polynomial")
  }
  if (kernel == "radial") {
    cat("Kernel selected as radial")
    model.SVM <- svm(as.factor(damage_grade) ~., data=trainFull, cost=cost,gamma=gamma, scale=TRUE, kernel="radial")
  }
  preds <- predict(model.SVM, valFull)
  acc <- mean(preds == valFull$damage_grade)
  cat('The accuracy for this SVM, with kernel', kernel, 'and cost', cost,'\n\n')
}
gammaSearch <- 10^(-9:3)
costSearch <- 10^(-3:3)
degreeSearch <- 1:5

library(parallelSVM)
model.SVM <- parallelSVM(as.factor(damage_grade) ~., data=trainFull, sampleSize=0.1,kernel="linear", cost=1)

preds <- predict(model.SVM, valFull)
acc <- mean(preds == valFull$damage_grade)

cat('The accuracy for this SVM, with kernel','linear', 'and cost', 0.5,'was',acc,'\n\n')

#linear search
for (c in c(0.01, 1, 10)){ 
  createSVMModel("linear", c, trainFull, valFull) 
}

#polynomial
polyParams <- expand.grid(costSearch, degreeSearch)
ntry <- 10
set.seed(5)
params <- sample(1:nrow(polyParams) , ntry, replace=FALSE)
for (i in params){
  c <- polyParams[i,]$cost
  deg <- polyParams[i,]$degree
  cat('PARAMETERS: COST',c,' DEGREE ',deg, '\n')
  createSVMModel("polynomial", c, trainFull, valFull, degree=deg) 
}

#radial
radParams <- expand.grid(cost=costSearch, gamme=gammaSearch)
ntry <- 10
set.seed(5)
params <- sample(1:nrow(radParams) , ntry, replace=FALSE)
for (i in params){
  c <- radParams[i,]$cost
  gam <- radParams[i,]$gamma
  createSVMModel("polynomial", c, trainFull, valFull, gamma=gam) 
  cat('PARAMETERS: COST',c,' GAMMA ',gam, '\n')
}
