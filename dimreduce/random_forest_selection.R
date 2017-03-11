library('randomForest')

csv <- read.csv('corpus.tfidf.104.csv')
frame <- data.frame(csv)


classes <- frame[,1]
matrix <- frame[,-1]

print('Building random forest')
forest <- randomForest(matrix, data=classes)
scores <- importance(forest)

print('----------------------------------------------------------')
print('Random Forest writing out to file')
write.csv(scores, file='randomforest.tfidf.105.csv')
