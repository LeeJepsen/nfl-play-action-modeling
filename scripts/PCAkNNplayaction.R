##NFL data
##Predict whether a Play Action pass will be
##successful based on game situations and defensive coverage schemes
##1. Performing PCA to transform game situations 
##2. defensive schemes & pass coverages as categories but transforming into
##dummy variables
##3. Applying KNN to classify Play Action success (binary outcomes)


##clear environment
rm(list = ls())

##Load packages
library(tidyverse)
library(caret)
library(gains)
library(pROC)
library(readxl)
library(class)


##load data
DF <- read_excel("C:/Users/YourName/Desktop/NFLplays.xlsx")

##Goal is to perform KNN on defensive characteristics and to incorporate
##PCA on targeted dimension reduction on game situation variables
##to reduce dimensions for KNN analysis

DF1 <- DF %>%
  filter(playAction == TRUE)

##PCA
##Creating separate feature for PCA
GameSituation <- DF1 %>%
  select(yardsToGo, absoluteYardlineNumber,
         preSnapHomeTeamWinProbability, preSnapVisitorTeamWinProbability,
         preSnapHomeScore, preSnapVisitorScore) %>%
  drop_na()

##Scale the all the numerical features
GameSituation <- scale(GameSituation)

##apply PCA
pcaGame <- prcomp(GameSituation)
pcaGame

##Checking varience and how much dimensionality to reduce
summary(pcaGame)

##based off of the summary and the eigenvalues to reduce dimensions and keep 
##98% of the variance I will keep the first four PCs 
##these 4 PCs retain 98% of the information from the game situation data

##Incorporate PC1-PC4 into dataset
pcaScores <- pcaGame$x

##into original DF
DF1 <- DF1 %>% 
  mutate(PC1game = pcaScores[, 1],
         PC2game = pcaScores[, 2],
         PC3game = pcaScores[, 3],
         PC4game = pcaScores[, 4])

##my umbrella interpretations of each PC score 
##PC1. Pre-snap win probability (Game Momentum)
##PC2. Scoring Differential
##PC3. Down and Distance
##PC4. Position on the field

##Since there would be 19 different levels to pass coverage I will reduce to 
##coverage based off function
unique(DF1$pff_passCoverage)

DF1$pff_passCoverage <- case_when(
  DF1$pff_passCoverage %in% c("Cover-3", "Cover-2", "Cover-1", 
                              "Quarters", "Cover 6-Left", 
                              "Cover 6-Right") ~ "Base Coverage",
  
  DF1$pff_passCoverage %in% c("Cover-3 Seam", "Cover-3 Double Cloud Bracket",
                              "Cover-3 Cloud Left", "Cover-3 Cloud Right",
                              "Bracket", "Cover-1 Double") ~ "Hybrid Coverage",
  
  DF1$pff_passCoverage %in% c("Goal Line", "Red Zone") ~ "Situational Coverage",
  
  DF1$pff_passCoverage %in% c("Prevent") ~ "Prevent Coverage",
  
  TRUE ~ "Unkown Coverage" ## This is NA & Miscellaneous
)

##double check everything lined up
unique(DF1$pff_passCoverage)

##Poor model results with multiple different variations to model
## so I changed the success metric of play action to amount gain on down
##instead of EPA, to be more specific
##40% gain on 1st down, 60% gain on 2nd down, 100%gain on 3rd/4th down
DF1 <- DF1 %>%
  mutate(playActionSuccess = case_when(
    down == 1 & yardsGained >= 0.4 * yardsToGo ~ 1,
    down == 2 & yardsGained >= 0.6 * yardsToGo ~ 1,
    down %in% c(3, 4) & yardsGained >= yardsToGo ~ 1,
    TRUE ~ 0
  ))

##check distribution
table(DF1$playActionSuccess)
prop.table(table(DF1$playActionSuccess))

##Convert pass coverage to factor, now with 5 levels
##Play Action success, and man/zone converted to factor
##Defining success as expected points added (EPA) to be greater than 1. 
##EPA = measure how much a single play is expected to contribute to a teams
##scoring potential based on current game situation (down,distance,field position),
##calculating difference between expected points before and after play occurs

DF1 <- DF1 %>%
  mutate(playAction = as.factor(ifelse(playAction == TRUE, 1, 0)),
         pff_passCoverage = as.factor(pff_passCoverage),
         pff_manZone = as.factor(pff_manZone),
         playActionSuccess = as.factor(playActionSuccess))


##One-hot encode pff_passCoverage and pff_manZone while keeping all levels
##same with down and quarter
##sifted through internet with this execution, couldnt keep
##all 4 variables for man/zone when initially trying, included lapply and fixed
DefensiveScemes <- model.matrix(~ pff_passCoverage + pff_manZone - 1, 
                                  data = DF1, 
                                  contrasts.arg = 
                                    lapply(DF1[c("pff_passCoverage", "pff_manZone")],
                                           contrasts, contrasts = FALSE)) %>% as.data.frame()


##Check column names
colnames(DefensiveScemes)


##Merge DefensiveSchemes, down, quarter, and PCA into KNN data set
DFkNN <- cbind(DF1 %>% select(PC1game, PC2game, PC3game, PC4game,
                              playActionSuccess), DefensiveScemes)

##double checking missing values
DFkNN <-drop_na(DFkNN)

##double checking data structure
str(DFkNN)

##splitting data for KNN into training and validation sets
##split 80:20
##tried 60,70,80,90.  80 had highest training accuracy
set.seed(27)
TrainIndex<- createDataPartition(DFkNN$playActionSuccess, p=0.8, list=FALSE)
TrainData <- DFkNN[TrainIndex,]
ValData <- DFkNN[-TrainIndex,]

##checking class distribution
table(TrainData$playActionSuccess)
table(ValData$playActionSuccess)

##Cross-Validation to help KNN model generalize well to new data
##help prevent overfitting
##In original try with EPA as success metric the following was true
##started with 10-fold CV, moved to 5 fold, with higher accuracy
##Train KNN with cross-validation **play action success**
##expand.grid trying k-values from 1-50, 40 seems to be the highest %acc ~50%

##By changing successful plays these are the parameters and results
##k-values 1:15, and cv at 5fold output the highest accuracy of 55%
set.seed(27)
KNNModel <- train(playActionSuccess ~.,
                  data = TrainData,
                  method = "knn",
                  trControl = trainControl(method = "cv", number = 5),
                  tuneGrid = expand.grid(.k = c(1:15)))

##view model summary
print(KNNModel)

##Apply validation set to evaluate performance
KNNPredictions <- predict(KNNModel, newdata = ValData)

##Confusion matrix
ConfusionMatrix <- confusionMatrix(KNNPredictions, ValData$playActionSuccess)
print(ConfusionMatrix)

##performing ROC curve to see if KNN provides any power in 
##probabilities and Decile-Wise lift chart to analyze model effectiveness 
##in decision-making

##predict probabilities instead of class labels
KNNprob <- predict(KNNModel, newdata = ValData, type = "prob")

##Compute
ROCcurve <- roc(ValData$playActionSuccess, KNNprob[,2])

##Plot
plot.roc(ROCcurve)
auc(ROCcurve)


##play action success back to numerical value
ValData$playActionSuccess <- as.numeric(as.character(ValData$playActionSuccess))
GainsTable <- gains(ValData$playActionSuccess, KNNprob[, 2])
GainsTable

##Decile-wise lift chart
barplot(GainsTable$mean.resp/mean(ValData$playActionSuccess), 
        names.arg=GainsTable$depth, 
        xlab="Percentile", 
        ylab="Lift", 
        ylim=c(0,3), 
        main="Decile-Wise Lift Chart")

##Results of model still struggled.. back to drawing board
##Going to try and look for more ways to utilize this technique, maybe try this model
##with classifying players as seen more common in sports