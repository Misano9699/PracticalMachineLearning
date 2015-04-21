# load libraries
library(caret);

# read training dataset and view the contents
pmlTraining <- read.csv("D:/workspace/Practical Machine Learning/Opdracht/Practical Machine Learning/pml-training.csv")
View(pmlTraining)

# do the same for the final test set
pmlTest <- read.csv("D:/workspace/Practical Machine Learning/Opdracht/Practical Machine Learning/pml-testing.csv")
View(pmlTest)

# view the number of rows and columns in the training set
dim(pmlTraining)
# view the head of the training set
head(pmlTraining)
# view the summary of the training set
summary(pmlTraining)

# we see a lot of columns with hardly any values, so use nearZeroVar to identify them
nearZeroVar(pmlTraining[,-160],saveMetrics=TRUE)

# remove the nzv columns from the training set
smallPml <- pmlTraining[,-nearZeroVar(pmlTraining[,-160])]

# remove all timing columns, because the test set only consists of a snapshot of an exercise, 
# so timing data is not relevant. And to be futureproof, I also don't want to use the username.
smallPml <- smallPml[,-c(1:6)]

dim(smallPml)
View(smallPml)

# remove all aggregate columns for the same reason
smallPml <- smallPml[,-c(grep("^max|^min|^var|^avg|^stddev|^amplitude", colnames(smallPml)))]

dim(smallPml)
View(smallPml)

# Now that we have simplified our training set, split this set in a training and test set to train and test our model.
# use 60% training vs 40% test (imho the training set is big enough)
inTrain <- createDataPartition(y=smallPml$classe, list=FALSE)
training <- smallPml[inTrain,]
testing <- smallPml[-inTrain,]

# use PCA to reduce the number of predictors, to gain 95% accuracy
pr<-preProcess(training[,-53], method="pca")

# train the model
trainPC <- predict(pr,training[,-53])
modelFit <- train(training$classe ~ .,method="rf",data=trainPC)
pr

# use this model on the testing data
testPC <- predict(pr, testing[,-53])
confusionMatrix(testing$classe, predict(modelFit, testPC))

# reduce the testset the same way as the as the training set
smallTest <- pmlTest[,-nearZeroVar(pmlTraining[,-160])]
smallTest <- smallTest[,-c(1:6)]
smallTest <- smallTest[,-c(grep("^max|^min|^var|^avg|^stddev|^amplitude", colnames(smallTest)))]
dim(smallTest)
View(smallTest)

# calculate the principal components for the final testset
finalPC <- predict(pr, smallTest[,-53])

# apply the model to the principal components
final <- predict(modelFit, finalPC)

# and view the results
final
