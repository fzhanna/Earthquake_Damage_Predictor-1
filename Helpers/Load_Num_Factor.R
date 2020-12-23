num <- function() {
  return(list(utils::read.csv('Preprocess_Analysis/Data/train_values_numeric.csv'), 
              utils::read.csv('Preprocess_Analysis/Data/train_labels_splitted.csv')))
}
factor <- function() {
  return(list(utils::read.csv('Preprocess_Analysis/Data/train_values_factor.csv'),
         utils::read.csv('Preprocess_Analysis/Data/train_labels_splitted.csv')))
}
