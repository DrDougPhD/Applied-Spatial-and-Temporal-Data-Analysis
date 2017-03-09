library('mRMRe')
set.thread.count(2)
csv <- read.csv('corpus.tfidf.105.csv')

#print('Classes')
#print(csv[,1])
#print('----------------------------------------------------------')
#print('Matrix')
#print(csv[,-1])
#print('----------------------------------------------------------')

frame <- data.frame(csv[,-1])
#classes <- frame[,1]
#matrix <- frame[,-1]

feature_data <- mRMR.data(data=frame)
print('----------------------------------------------------------')
print('Writing out Mutual Information Matrix')
write.csv(mim(feature_data), file='coinformation.tfidf.105.csv')

#data(cgps)
#data.annot <- data.frame(cgps.annot)
#data.cgps <- data.frame(cgps.ic50, cgps.ge)
#print(data.cgps)

#data(cgps)
#mRMRdata <- mRMR.data(data = data.frame(target=cgps.ic50, cgps.ge))
#dd <- subsetData(mRMRdata, 1:10, 1:10)
#mRMR.ensemble(data = data, target_indices = 1, 
#              feature_count = 30, solution_count = 1)
#print(mim(dd))