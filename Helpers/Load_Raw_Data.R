trainVal <- function() {
  return(utils::read.csv('Raw_Data/train_values.csv'))
}
testVal <- function() {
  return(utils::read.csv('Raw_Data/test_values.csv'))
}
trainLab <- function() {
  return(utils::read.csv('Raw_Data/train_labels.csv'))
}
