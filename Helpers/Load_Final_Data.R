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
