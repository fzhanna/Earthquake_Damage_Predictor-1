loadTrain <- function() {
  return(list(utils::read.csv('Preprocess_Analysis/Data/train_values_splitted.csv'),
              utils::read.csv('Preprocess_Analysis/Data/train_labels_splitted.csv')))
}

loadVal <- function() {
  return(list(utils::read.csv('Preprocess_Analysis/Data/val_values_splitted.csv'),
              utils::read.csv('Preprocess_Analysis/Data/val_labels_splitted.csv')))
}