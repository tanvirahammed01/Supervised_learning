#Project Code
#Data Management and EDA
#importing data into R
rm(list=ls())

#setwd("D:\\M2\\Mining\\Project\\archive")
estonia <- read.csv("estonia-passenger-list.csv", header= T)
str(estonia)
estonia = subset(estonia, select = -c(PassengerId, Firstname, Lastname))
#preparing the variables
estonia$Country= ifelse(estonia$Country == "Sweden", 0, ifelse(estonia$Country == "Estonia", 1, 2))
estonia$Country = factor(estonia$Country,
                         levels = c(0,1,2),
                         labels = c('Sweden', 'Estonia', 'Other'))
estonia$Sex = ifelse(estonia$Sex=='M', 1, 0)
estonia$Sex = factor(estonia$Sex,
                     levels = c(0, 1),
                     labels = c('Female', 'Male'))
estonia$Age = ifelse(estonia$Age < 18, 0, ifelse( estonia$Age > 17 & estonia$Age < 60, 1, 2))
estonia$Age = factor(estonia$Age,
                     levels = c(0,1,2),
                     labels = c('Minor', 'Adult', 'Senior Adult'))
estonia$Category = ifelse(estonia$Category== 'C', 1, 0)
estonia$Category = factor(estonia$Category,
                          levels = c(0,1),
                          labels = c('Passenger','Crew'))
estonia$Survived = factor(estonia$Survived,
                          levels = c(0,1),
                          labels = c('Died', 'Survived'))
#Frequency distribution according to passengers’ characteristics
par(mfrow=c(2,2))
barplot(table(estonia$Sex),
        main = "Frequency distribution of the passengers' gender", )
barplot(table(estonia$Age),
        main = "Frequency distribution of the passengers' age", )
barplot(table(estonia$Country),
        main = "Frequency distribution of the passengers' country", )
barplot(table(estonia$Category),
        main = "Frequency distribution of the passengers' category", )
#barplot: survival by category
library(GGally)
ggbivariate(data = estonia, outcome = "Survived",  explanatory = c('Age' , "Sex", "Category", 'Country'))

#Machine learning techniques
#logistic regression
# Create Training Data and Test Data
set.seed(100)
samp = sample.int(n=nrow(estonia), size = floor(.7*nrow(estonia)), replace = F)
train = estonia[samp,]
test = estonia[-samp,]
#undersampling and oversampling on this imbalanced data
library(ROSE)
data_balanced_both <- ovun.sample(Survived ~ ., data = train, method = "both", p=0.5, N=1000, seed = 1)$data
#build logistic regression model
lr.rose <- glm(Survived ~ ., data = data_balanced_both, family = binomial)
#make predictions on unseen data
prediction = plogis(predict(lr.rose, newdata = test))
prediction = round(prediction, digits = 0)
#accuracy rate
tab = table(test$Survived, prediction)
sum(diag(tab))/sum(tab)
#Confusion matrix
#install.packages('yardstic')
library(yardstick)
library(ggplot2)
truth_predicted <- data.frame(test$Survived, prediction)
truth_predicted$prediction <- as.factor(truth_predicted$prediction)


truth_predicted$test.Survived<- ifelse(truth_predicted$test.Survived == "Died", 0, 1)
truth_predicted$test.Survived <- as.factor(truth_predicted$test.Survived)

lrcm <- conf_mat(truth_predicted, test.Survived, prediction )
autoplot(lrcm, type = "heatmap")
#odds ratio
exp(coef(lr.rose))
install.packages('sjPlot')
library(sjPlot)
plot_model(lr.rose, show.values = TRUE, title = "")


#KNN
library(class)
rm(list=ls())



#setwd("D:\\M2\\Mining\\Project\\archive")
estonia <- read.csv("estonia-passenger-list.csv", header= T)
#Data Management
estonia$Country= ifelse(estonia$Country == "Sweden", 0, ifelse(estonia$Country == "Estonia", 1, 2))
estonia$Country = as.numeric(estonia$Country,
                             levels = c(0,1,2),
                             labels = c('Sweden', 'Estonia', 'Other'))
estonia$Sex = ifelse(estonia$Sex=='M', 0, 1)
estonia$Sex = as.numeric(estonia$Sex,
                         levels = c(1,0),
                         labels = c('Female','Male'))
estonia$Age = ifelse(estonia$Age < 18, 0, ifelse( estonia$Age > 17 & estonia$Age < 60, 1, 2))
estonia$Age = as.numeric(estonia$Age,
                         levels = c(0,1,2),
                         labels = c('Minor', 'Adult', 'Senior Adult'))
estonia$Category = ifelse(estonia$Category== 'C', 0, 1)
estonia$Category = as.numeric(estonia$Category,
                              levels = c(1,0),
                              labels = c('Passenger','Crew'))
estonia$Survived = as.factor(estonia$Survived)
estonia = subset(estonia, select = -c(PassengerId, Firstname, Lastname))
# Create Training Data and Test Data
set.seed(100)
samp = sample.int(n=nrow(estonia), size = floor(.7*nrow(estonia)), replace = F)
train = estonia[samp,]
test = estonia[-samp,]
#Find the number of observation
sqrt (NROW(train))
# The square root of 700 is around 26.31, therefore we’ll create a model
#with a ‘K’ value as 27.
set.seed(1)
knn.27 <- knn(train, test, train$Survived, k=27)
#Calculate the proportion of correct classification for k = 26, 27
tab = table(test$Survived, knn.27)
sum(diag(tab))/sum(tab)#Confusion matrix
truth_predicted <- data.frame(test$Survived, knn.27)
knncm <- conf_mat(truth_predicted, test.Survived, knn.27 )
autoplot(knncm, type = "heatmap")
#SVM
str(train)
library(ROSE)
#generate data synthetically
#install.packages('ROSE')
data.rose <- ROSE(Survived ~ ., data = train, seed = 1)$data

install.packages('e1071')
library(e1071)
#selecting the parameters
tune.out <- tune(svm, Survived~., data = data.rose, ranges = list(kernel= c("linear", "polynomial", "radial", "sigmoid"), gamma = 2^(-1:1), cost
                                                                  = (0.001*10^(0:5))))
summary(tune.out)
# extract the best model
(bestmod <- tune.out$best.model)
# validate model performance
(out <- svm(Survived~., data = data.rose, gamma = 0.5, kernel = "linear",
            cost=0.001))
pred.test <- predict(out, newdata=test)
(tab<-table(pred.test, test$Survived))
#accuracy
sum(diag(tab))/sum(tab)
#Confusion matrix
truth_predicted <- data.frame(test$Survived, pred.test)
svmcm <- conf_mat(truth_predicted, test.Survived, pred.test)
autoplot(svmcm, type = "heatmap")


#RF
install.packages('caret')
library(caret)
# Define the control
trControl <- trainControl(method = "cv", number = 10, search = "grid")
rf_default <- train(Survived~., data = data.rose, method = "rf", metric =
                      "Accuracy", trControl = trControl)
# Print the results
print(rf_default)
set.seed(1234)
#finding the best mtry value
tuneGrid <- expand.grid(.mtry = c(4: 10))
rf_mtry <- train(Survived~., data = data.rose, method = "rf", metric = "A
ccuracy", tuneGrid = tuneGrid, trControl = trControl, importance = TRUE,
nodesize = 14, ntree = 300)
print(rf_mtry)
#storing the best mtry value
best_mtry <- rf_mtry$bestTune$mtry
tuneGrid <- expand.grid(.mtry = best_mtry)

# Search the best maxnodes
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(5: 15)) {
  set.seed(1234)
  rf_maxnode <- train(Survived~., data = data.rose, method = "rf", metric
                      = "Accuracy", tuneGrid = tuneGrid, trControl = trControl, importance = TRUE, nodesize = 14, maxnodes = maxnodes, ntree = 300)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)
#The highest accuracy score is obtained with a value of maxnode equals to
7.
#Search the best ntrees
store_maxtrees <- list()
for (ntree in c(100, 250, 300, 350, 400, 450, 500, 550)) {
  set.seed(5678)
  rf_maxtrees <- train(Survived~., data = data.rose, method = "rf", metric = "Accuracy", tuneGrid = tuneGrid, trControl = trControl, importance =
                         TRUE, nodesize = 14, maxnodes = 7, ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)
#final model
fit_rf <- train(Survived~., data.rose, method = "rf", metric = "Accuracy"
                , tuneGrid = tuneGrid, trControl = trControl, importance = TRUE, ntree =
                  300, maxnodes = 7)
#accuracy
prediction <-predict(fit_rf, test)
confusionMatrix(prediction, test$Survived)
#Confusion matrix
truth_predicted <- data.frame(test$Survived, prediction)
rfcm <- conf_mat(truth_predicted, test.Survived, prediction)
autoplot(rfcm, type = "heatmap")


#Tree
#install.packages('tree')
library(tree)
attach(train)
#over and under sampling
data_balanced_both <- ovun.sample(Survived ~ ., data = train, method = "both", p=0.5, N=989, seed = 1)$data
#building the model
treeM = tree(Survived ~., data_balanced_both)
summary(treeM)
plot(treeM)
text(treeM, pretty=0)
#accuracy
tree.pred = predict(treeM, test, type="class")
tab<-table(tree.pred, test$Survived)
sum(diag(tab))/sum(tab)
#Confusion matrix
truth_predicted <- data.frame(test$Survived, tree.pred)
tcm <- conf_mat(truth_predicted, test.Survived, tree.pred)
autoplot(tcm, type = "heatmap")



#ANN
estonia$Survived = as.numeric(estonia$Survived)
#estonia = subset(estonia, select = -c(PassengerId, Firstname, Lastname))
# Create Training Data and Test Data
set.seed(100)
samp = sample.int(n=nrow(estonia), size = floor(.7*nrow(estonia)), replace = F)
train = estonia[samp,]
test = estonia[-samp,]
data.rose <- ROSE(Survived ~ ., data = train, seed = 1)$data
install.packages('neuralnet')
library(neuralnet)
# 1-Hidden Layer, 1-neuron
set.seed(100)
attach(data.rose)
model = Survived ~.
ANNM1 <- neuralnet(model, data=data.rose, linear.output = FALSE, likelihood = TRUE)
# 1-Hidden Layer, 2-neuron
set.seed(100)
ANNM2 <- neuralnet(model, data=data.rose, linear.output = FALSE, likelihood = TRUE, hidden = 2)
# 1-Hidden Layer, 3-neuron
set.seed(100)
ANNM3 <- neuralnet(model, data=data.rose, linear.output = FALSE, likelihood = TRUE, hidden = 3)
# 1-Hidden Layer, 4-neuron
set.seed(100)
ANNM4<- neuralnet(model, data=data.rose, linear.output = FALSE, likelihood = TRUE, hidden = 4)
# 1-Hidden Layer, 5-neuron
set.seed(100)
ANNM5<- neuralnet(model, data=data.rose, linear.output = FALSE, likelihood= TRUE, hidden = 5)
# 2-Hidden Layer, 3-neuron
set.seed(100)
ANNM6<- neuralnet(model, data=data.rose, linear.output = FALSE, likelihood = TRUE, hidden = c(3,3))
temp_test<- subset(test, select = c('Age', "Sex", 'Country', 'Category'))
ANNM.results <- compute(ANNM3, temp_test)
results <- data.frame(actual = test$Survived, prediction = ANNM.results$net.result)
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
#Accuracy rate
attach(roundedresultsdf)
print(tab<-table(actual, prediction))
sum(diag(tab))/sum(tab)

#Confusion matrix
truth_predicted <- data.frame(test$Survived, roundedresultsdf$prediction)
truth_predicted$test.Survived= as.factor(truth_predicted$test.Survived)
truth_predicted$roundedresultsdf.prediction = factor(truth_predicted$roundedresultsdf.prediction, levels=c(1,2))


anncm <- conf_mat(truth_predicted, test.Survived, roundedresultsdf.prediction)
autoplot(anncm, type = "heatmap")
