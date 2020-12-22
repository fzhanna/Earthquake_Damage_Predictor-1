trainVal <- function(seed, prob, fullValues, fullLabels) {
  valIndex <- caret::createDataPartition(fullLabels[,1], p=prob, 
                                          list = FALSE, times = 1)
  return(list(fullValues[-valIndex,], fullLabels[-valIndex,], 
         fullValues[valIndex,], fullLabels[valIndex,]))
}

