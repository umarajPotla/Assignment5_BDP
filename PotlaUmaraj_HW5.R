library(caret)
library(gbm)
library(e1071)

data(scat)

str(scat)

# 1. Set the Species column as the target/outcome and convert it to numeric.
outcomeName<-'Species'

#Converting outcome variable to numeric
scat$Species<-ifelse(scat$Species=="bobcat",0,ifelse(scat$Species=="coyote", 1, 2))

# After converting outcome variable to numeric
str(scat)

# 2. Remove the Month, Year, Site, Location features.
scat[ ,c('Month', 'Year', 'Site', 'Location')] <- list(NULL)
str(scat)

# 3. Check if any values are null. 
# If there are, impute missing values using KNN.
sum(is.na(scat))
preProcValues <- preProcess(scat[,c('Age', 'Number', 'Length', 'Diameter','Taper','TI','Mass','d13C','d15N','CN','ropey','segmented','flat','scrape')], method = c("knnImpute","center","scale"))

library('RANN')
scat_processed <- predict(preProcValues, scat)
sum(is.na(scat_processed))

# 4. Converting every categorical variable to numerical (if needed).

str(scat_processed) # after processing

# Answer - After removing columns in Question 2 and converting outcome column
# to numeric, there are no more categorical variables. 
# There is no need for any conversion at this point.

# 5. With a seed of 100, 75% training, 25% testing. 
# Build the following models: randomforest, neural net, naive bayes and GBM.
# a. For these models display a)model summarization and 
# b) plot variable of importance, for the predictions (use the prediction set) display 
# c) confusion matrix (60 points)

#Converting the dependent variable back to categorical
scat_processed$Species<-as.factor(scat_processed$Species)

str(scat_processed)

#Spliting training set into two parts based on outcome: 75% and 25%
set.seed(100)
index <- createDataPartition(scat_processed$Species, p=0.75, list=FALSE)
trainSet <- scat_processed[ index,]
testSet <- scat_processed[-index,]

#Checking the structure of trainSet
str(trainSet)

# all variables as predictors
predictors<-c("Age", "Number", "Length", "Diameter", "Taper", "TI", "Mass", "d13C", 
              "d15N", "CN", "ropey", "segmented", "flat", "scrape")


######## Training Models Using Caret ############
names(getModelInfo())

# ##### randomforest ######
model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf', importance=T)
# summarizing the model
print(model_rf)
# Visualizing the models
plot(model_rf)
#Variable Importance
varImp(object=model_rf)
#Plotting Varianle importance for Random Forest
plot(varImp(object=model_rf),main="Random Forest - Variable Importance")
#Predictions
predictions_rf<-predict.train(object=model_rf,testSet[,predictors],type="raw")
table(predictions_rf)
#Confusion Matrix and Statistics
confusionMatrix(predictions_rf,testSet[,outcomeName])

# ######## Neural Net #######
model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet', importance=T)
# summarizing the model
print(model_nnet)
# Visualizing the models
plot(model_nnet)
#Variable Importance
nnet_varImp<- varImp(object=model_nnet)
#Plotting Variable importance for Neural Net
plot(varImp(object=model_nnet),main="Neural Net - Variable Importance")
#Predictions
predictions_nnet<-predict.train(object=model_nnet,testSet[,predictors],type="raw")
table(predictions_nnet)
#Confusion Matrix and Statistics
confusionMatrix(predictions_nnet,testSet[,outcomeName])

# ########### Naive Bayes ###########
model_nb<-train(trainSet[,predictors],trainSet[,outcomeName],method='naive_bayes', importance=T)
# summarizing the model
print(model_nb)
# Visualizing the models
plot(model_nb)
#Variable Importance
varImp(object=model_nb)
#Plotting Variable importance for Naive Bayes
plot(varImp(object=model_nb),main="Naive Bayes - Variable Importance")
#Predictions
predictions_nb<-predict.train(object=model_nb,testSet[,predictors],type="raw")
table(predictions_nb)
#Confusion Matrix and Statistics
confusionMatrix(predictions_nb,testSet[,outcomeName])

# ############ GBM ##########
model_gbm1<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')
# summarizing the model
print(model_gbm1)
# Visualizing the models
plot(model_gbm1)
#Variable Importance
varImp(object=model_gbm1)
#Plotting Variable importance for GBM
plot(varImp(object=model_gbm1),main="GBM - Variable Importance")
#Predictions
predictions_gbm<-predict.train(object=model_gbm1,testSet[,predictors],type="raw")
table(predictions_gbm)
#Confusion Matrix and Statistics
confusionMatrix(predictions_gbm,testSet[,outcomeName])

# 6. For the BEST performing models of each (randomforest, neural net, naive bayes and gbm) 
# create and display a data frame that has the following columns: 
# ExperimentName, accuracy, kappa. 
# Sort the data frame by accuracy. 

experimentName<-c("Random Forest" , "Neural Net", "Naive Bayes",  "GBM")
accuracyDetails<-c(max(model_rf$results$Accuracy), max(model_nnet$results$Accuracy), 
                   max(model_nb$results$Accuracy), max(model_gbm1$results$Accuracy))

kappaDetails<-c(max(model_rf$results$Kappa), max(model_nnet$results$Kappa), 
                   max(model_nb$results$Kappa), max(model_gbm1$results$Kappa))

bestModelDf<-data.frame(ExperimentName=experimentName, Accuracy=accuracyDetails, Kappa=kappaDetails)
print(bestModelDf[order(-bestModelDf$Accuracy),])

# 7. Tune the GBM model using tune length = 20 and: 
# a) print the model summary and b) plot the models. (20 points)
#using tune length
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)

### Using tuneGrid ####
modelLookup(model='gbm')

model_gbm2<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneLength=20)
# a) print the model summary
print(model_gbm2)
#b) plot the models. 
plot(model_gbm2)

# 8. Using GGplot and gridExtra to plot all variable of 
# importance plots into one single plot. (10 points)
library(ggplot2)
library(gridExtra)

plot_gbm1 <- ggplot(data=varImp(object=model_gbm1)) + ggtitle("GBM")
print(plot_gbm1)
plot_nb <- ggplot(data=varImp(object=model_nb)) + ggtitle("Naive Bayes")
print(plot_nb)
plot_nnet <- ggplot(data=varImp(object=model_nnet)) + ggtitle("Neural Network")
print(plot_nnet)
plot_rf <- ggplot(data=varImp(object=model_rf)) + ggtitle("Random Forest")
print(plot_rf)

grid.arrange(plot_gbm1, plot_nb, plot_rf, nrow = 2, ncol=2, heights = c(0.35, 0.65))


# 9. Which model performs the best? and why do you think this is the case? 
# Can we accurately predict species on this dataset? (10 points)

# Answer - 
# As per accuracy & Kappa values of the models, Neural Network model performs the best. 
#  ExperimentName   Accuracy        Kappa
# 2     Neural Net  0.6943589   0.5100720
# 1  Random Forest  0.6479826   0.4170319
# 3    Naive Bayes  0.6461044   0.4180012
# 4            GBM  0.6155064   0.3710730
# Neural Network model uses memory to store previous read values and 
# does classification based on correlation between the read values.
# Whereas, other models like Naive Bayes considers attributes are independent of each other.
# Species from the data set can be predicted with less than 70 % accuracy 
# i.e. 69.4% using Neural Networks model, 64.8% using Random Forest model can be accurately predicted.



# 10. Graduate Student questions:
# a. Using feature selection with rfe in caret and the repeatedcv method: Find the top 3
# predictors and build the same models as in 6 and 8 with the same parameters. (20 points)

#Feature selection using rfe in caret
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)

predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]
Species_Pred <- rfe(trainSet[,predictors], trainSet[,outcomeName],rfeControl = control)
Species_Pred

#output was :
# The top 4 variables (out of 4):
# CN, d13C, d15N, Mass

#Taking only the top 3 predictors
predictorsTop3<-c("CN", "d13C", "d15N")

# ######## randomforest  with top-3 selected features ###########
model_rf_new<-train(trainSet[,predictorsTop3],trainSet[,outcomeName],method='rf', importance=T)
# summarizing the model
print(model_rf_new)
# Visualizing the models
plot(model_rf_new)
#Variable Importance
varImp(object=model_rf_new)
#Plotting Varianle importance for Random Forest
plot(varImp(object=model_rf_new),main="Random Forest - Variable Importance")
#Predictions
predictions_rf_new<-predict.train(object=model_rf_new,testSet[,predictorsTop3],type="raw")
table(predictions_rf_new)
#Confusion Matrix and Statistics
confusionMatrix(predictions_rf_new,testSet[,outcomeName])

# ####### Neural Net with top-3 selected features ###########
model_nnet_new<-train(trainSet[,predictorsTop3],trainSet[,outcomeName],method='nnet', importance=T)
# summarizing the model
print(model_nnet_new)
# Visualizing the models
plot(model_nnet_new)
#Variable Importance
varImp(object=model_nnet_new)
#Plotting Variable importance for Neural Net
plot(varImp(object=model_nnet_new),main="Neural Net - Variable Importance")
#Predictions
predictions_nnet_new<-predict.train(object=model_nnet_new,testSet[,predictorsTop3],type="raw")
table(predictions_nnet_new)
#Confusion Matrix and Statistics
confusionMatrix(predictions_nnet_new,testSet[,outcomeName])

# ############# Naive Bayes with top-3 selected features ###########
model_nb_new<-train(trainSet[,predictorsTop3],trainSet[,outcomeName],method='naive_bayes', importance=T)
# summarizing the model
print(model_nb_new)
# Visualizing the models
plot(model_nb_new)
#Variable Importance
varImp(object=model_nb_new)
#Plotting Variable importance for Naive Bayes
plot(varImp(object=model_nb_new),main="Naive Bayes - Variable Importance")
#Predictions
predictions_nb_new<-predict.train(object=model_nb_new,testSet[,predictorsTop3],type="raw")
table(predictions_nb_new)
#Confusion Matrix and Statistics
confusionMatrix(predictions_nb_new,testSet[,outcomeName])

# ############### GBM with top-3 selected features ###########
model_gbm1_new<-train(trainSet[,predictorsTop3],trainSet[,outcomeName],method='gbm')

# summarizing the model
print(model_gbm1_new)
# Visualizing the models
plot(model_gbm1_new)
#Variable Importance
varImp(object=model_gbm1_new)
#Plotting Variable importance for GBM
plot(varImp(object=model_gbm1_new),main="GBM - Variable Importance")
#Predictions
predictions_gbm_new<-predict.train(object=model_gbm1_new,testSet[,predictorsTop3],type="raw")
table(predictions_gbm_new)
#Confusion Matrix and Statistics
confusionMatrix(predictions_gbm_new,testSet[,outcomeName])

# 10. b. Create a dataframe that compares the non-feature selected models ( the same as on 7)
# and add the best BEST performing models of each (randomforest, neural net, naive bayes and gbm) and 
# display the data frame that has the following columns: ExperimentName, accuracy, kappa. 
# Sort the data frame by accuracy. (40 points)

experimentName_new<-c("Random Forest New" , "Neural Net New", "Naive Bayes New",  "GBM New")
accuracyDetails_new<-c(max(model_rf_new$results$Accuracy), max(model_nnet_new$results$Accuracy), 
                   max(model_nb_new$results$Accuracy), max(model_gbm1_new$results$Accuracy))

kappaDetails_new<-c(max(model_rf_new$results$Kappa), max(model_nnet_new$results$Kappa), 
                max(model_nb_new$results$Kappa), max(model_gbm1_new$results$Kappa))

bestModelDf_new<-data.frame(ExperimentName=c(experimentName, experimentName_new), 
                            Accuracy=c(accuracyDetails, accuracyDetails_new), 
                            Kappa=c(kappaDetails, kappaDetails_new))
print(bestModelDf_new[order(-bestModelDf_new$Accuracy),])

# NOTE:::  "New" refers to the latest models with Top 3 predictors in 
# "Random Forest New" , "Neural Net New", "Naive Bayes New",  "GBM New".

# c. Which model performs the best? and why do you think this is the case? 
# Can we accurately predict species on this dataset? (10 points)

# Answer - 
# Output of Best Model dataframe :::
#      ExperimentName   Accuracy        Kappa
# 6    Neural Net New   0.7522766   0.5693890
# 7   Naive Bayes New   0.7254481   0.5135670
# 2        Neural Net   0.6943589   0.5100720
# 5 Random Forest New   0.6861781   0.4607807
# 1     Random Forest   0.6479826   0.4170319
# 3       Naive Bayes   0.6461044   0.4180012
# 8           GBM New   0.6352902   0.3809075
# 4               GBM   0.6155064   0.3710730

# From the Best model printed above with based on accuracy, it is evident that Neural Network New
# (with top 3 predictors) performed the best with an accuracy of 75.22% and Kappa value of 56.9%. 
# Followed by Naive Bayes(with top 3 predictors) and the old Neural network model (with non-feature selected)
# with accuracies of 72.54 & 69.43 % respectively.
# With the new Neural Network model with top 3 selected features, prediction accuracy increased from 69.4 % to 75.22 %.
# Prediction using this model is improved. 
# Yes we can predict Species from the dataset using Neural Network New model with 75.22% accuracy.


