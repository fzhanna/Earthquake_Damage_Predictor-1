combineLab <- function(data, label) {
  return(dplyr::inner_join(data, label))
}