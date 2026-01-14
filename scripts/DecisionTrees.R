##NFL pre snap data
##Cleaning and organizing data for Supervised Data mining techniques
##Goal is to predict whether a play with be play-action pass
##based on pre-snap alignment and game variables
##Intended modeling is decision trees

##clear working environment
rm(list=ls())

##Load libraries
library(readxl)
library(dplyr)
library(ggplot2)
library(randomForest)
library(adabag)
library(gains)
library(caret)
library(rpart)
library(rpart.plot)
library(pROC)

##Bringing in data
df <- read_excel("C:/Users/YourName/Desktop/NFLplays.xlsx")

##data comes clean of any special teams plays, QB spikes/kneels no need to filter out
##selecting data for analysis and model
df1 <- df %>%
  select(
    playAction,                                     ##this is target variable
    receiverAlignment,                              ##context to offensive formation
    down, yardsToGo,                                ##pre-snap context
    quarter,                                        ##game management
    pff_passCoverage, pff_manZone,                  ##Defensive Scheme
    pff_runPassOption,                              ##RPO indicator
    ##Probability that the team will win the game before the snap,
    ##takes into account many game factors before the play,
    ##timeouts remaining, down, distance, score differential, and few others
    ##I thought it would be a good variable to throw into models
    preSnapHomeTeamWinProbability, preSnapVisitorTeamWinProbability
    )

##Converting categorical variables to factors for decision trees
df2 <- df1 %>%
  mutate(
    playAction = as.factor(playAction),
    receiverAlignment = as.factor(receiverAlignment),
    down = as.factor(down),
    pff_passCoverage = as.factor(pff_passCoverage),
    pff_manZone = as.factor(pff_manZone),
    pff_runPassOption = as.factor(pff_runPassOption)
  )


##Exploring data
##double checking data structures and summary before decision trees
str(df2)

##checking the balance of plays actions vs non play actions so I can try to 
##evaluating imbalance
table(df2$playAction)
prop.table(table(df2$playAction)) *100

##Create a dataframe from the proportions
playactiondf <- data.frame(
  PlayAction = c("FALSE", "TRUE"),
  Percentage = prop.table(table(df2$playAction)) * 100
)

# Plot the bar graph
ggplot(playactiondf, aes(x = PlayAction, y = Percentage.Freq, fill = PlayAction)) +
  geom_bar(stat = "identity") +
  labs(title = "Play Action Percentage", x = "Play Action", y = "Percentage of Total Plays") +
  theme_minimal() +
  scale_fill_manual(values = c("FALSE" = "darkblue", "TRUE" = "lightgreen"))

##data is imbalanced, making some visualizations get an idea of distribution
##right away Im thinking of leaning into gini impurity index, ensemble decision trees, 
##incorporating some class weight to even out the proportion

##Play action frequency by down
ggplot(df2, aes(x = down, fill = playAction)) +
  geom_bar(position = "fill") +
  labs(title = "play Action Usage by Down", y = "Proportion", x = "Down") +
  theme_minimal()

##Receiver alignment during play action
ggplot(df2, aes(x = receiverAlignment, fill = playAction)) +
  geom_bar(position = "fill") +
  labs(title = "Play Action rate by receiver alignment", y = "Proportion",
       x = "Receiver Alignment") + 
  theme(axis.title.x = element_text(angle = 0, hjust = 1))

##Play action usage by yards to go
ggplot(df2, aes(x = yardsToGo, fill = playAction)) +
  geom_histogram(binwidth = 3, position = "fill") + 
  labs(title = "Play Action Usage by Yards to Go", y = "Proportion", x = "Yards to Go") + 
  theme_minimal()

##Defensive scheme vs Play Action usage
ggplot(df2, aes(x = pff_passCoverage, fill = playAction)) + 
  geom_bar(position = "fill") +
  labs(title = "Play Action Usage by Defensive Coverage", y = "Proportion", 
       x = "Coverage Type") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

##do winning teams use play action more/less
##Play action vs pre-snap Win probability
ggplot(df2, aes(x = preSnapHomeTeamWinProbability, fill = playAction)) +
  geom_density(alpha = 0.5) + 
  labs(title = "Play Action Usage vs Pre-Snap Win Probability", x = "Home Team Win Probabilty") +
  theme_minimal()


##After exploring the data
##Stating 1st and 2nd down, then grouping 3rd and 4th since there was a large drop off
##in usage on those downs
df3 <- df2 %>%
  mutate(down = case_when(
    down == 1 ~ "1st Down",
    down == 2 ~ "2nd Down",
    down %in% c(3, 4) ~ "Late Downs"
  ))

##change back to factor
df3$down <- factor(df3$down, levels = c("1st Down", "2nd Down", "Late Downs"))

##visualize after grouping
ggplot(df3, aes(x = down, fill = playAction)) +
  geom_bar(position = "fill") +
  labs(title = "Play Action Usage by down Category", y = "Proportion", x = "down Category") +
  theme_minimal()


## Grouping the types of receiver alignments
df3 <- df2 %>%
  mutate(
    receiverAlignment = case_when(
      receiverAlignment %in% c("1x0", "1x1", "2x1") ~ "Tight",
      receiverAlignment %in% c("2x2", "3x0", "3x1") ~ "Balanced",
      receiverAlignment %in% c("3x2", "4x1", "4x2") ~ "Spread",
      TRUE ~ "Unknown"  
    )
  )

##back to a factor
df3$receiverAlignment <- factor(df3$receiverAlignment, levels = c("Tight", "Balanced", "Spread"))

##Visualize change
ggplot(df3, aes(x = receiverAlignment, fill = playAction)) +
  geom_bar(position = "fill") +
  labs(title = "Play Action rate by receiver alignment", y = "Proportion",
       x = "Receiver Alignment") + 
  theme(axis.title.x = element_text(angle = 0, hjust = 1))


##Binning yards to go into 3 categories short, medium, and long
df3 <- df2 %>%
  mutate(yardsToGo = case_when(
    yardsToGo <= 3 ~ "Short",
    yardsToGo <= 9 ~ "Medium",
    yardsToGo >= 10 ~ "Long"
  ))

##back to a factor
df3$yardsToGo <- factor(df3$yardsToGo, levels = c("Short", "Medium", "Long"))

##Visualize change
ggplot(df3, aes(x = yardsToGo, fill = playAction)) +
  geom_bar(position = "fill") + 
  labs(title = "Play Action Usage by Yards to Go", y = "Proportion", x = "Yards to Go") + 
  theme_minimal()


#########
##exploring data with decision trees
##partition data into validation and training set
##random seed
set.seed(27)
##partitioning to 60:40
##changing to 60:40 from a 70:30 or 80:20 decreased the accuracy by a 0.009 
##but increased specificity by 0.02
##lost accuracy but increases predictability of true negatives
myIndex <- createDataPartition(df3$playAction, p=0.6, list=FALSE)
##splitting the data
TrainData <- df3[myIndex,]
ValData <- df3[-myIndex,]

##Check distribution
table(TrainData$playAction)
table(ValData$playAction)

##Train Decision Tree on training data
##made many changes to variables included and none that successfully displayed a lower
##xerror score than 1
##including Gini index
TreeModel <- rpart(playAction ~ yardsToGo + down + quarter + pff_manZone + pff_passCoverage +
                     pff_runPassOption,
                   data = TrainData, 
                   method = "class",
                   parms = list(split = "gini"))

##Summary of the Decision Tree 
summary(TreeModel)

##Visualize the full tree
set.seed(27)
FullTree <- rpart(playAction ~ yardsToGo + down + quarter + pff_manZone + pff_passCoverage +
                    pff_runPassOption, 
                   data = TrainData, 
                   method = "class", 
                   cp = 0, 
                   minsplit = 2, 
                   minbucket = 1)
prp(FullTree, 
    type = 1, 
    extra = 1, 
    under = TRUE)

printcp(FullTree)


##I do not get a decrease in value but rather an increase in xerror starting at the first tree
##starting at 1.00, class imbalance which leads to overfitting 
##because of these results I cannot prune the tree
##After seeing results of regular decision tree, I will opt to try a random forest decision tree
##since I have a fairly uneven balance 17%/83%
##Reducing the weight for play action, because model may overcompensate for the imbalance
##after multiple runs of this and exploring I introduced a weight 0.5,and changed number of trees
##to train on to 100 which slightly increased specificity because it was extremely low
RFmodel <- randomForest(playAction ~ pff_runPassOption + quarter + 
                          pff_manZone + down + pff_passCoverage,
                        data = TrainData,
                        ntree = 100,
                        mtry = 2,
                        classwt = c("FALSE" = 1, "TRUE" = 0.5),
                        importance = TRUE)

##importance of each variable
##after multiple tweaking for model performance, variable: yards to go, negatively impacted
##model by -10 mean decrease accuracy, I dropped that variable
##dropped both home and visitor team percentage to win because it was over powering
##decision making of the training model. Indicating both are highly correlated variables
##as teams are winning the amount of play action usage increases
varImpPlot(RFmodel, type = 1)

##predict on test
RFpredictions <- predict(RFmodel, ValData)

##evaluate performance
confusionMatrix(RFpredictions, ValData$playAction)

##predict the probability of a Play Action pass
PredictedProb <- predict(RFmodel, ValData, type= 'prob')

##convert to a numeric
ValData$playAction <- as.numeric(as.factor(ValData$playAction)) - 1

##create gains table
GainsTable <- gains(ValData$playAction, PredictedProb [,2])
GainsTable

##create receiver operator curve
RocCurve <- roc(ValData$playAction, PredictedProb[,2])
plot.roc(RocCurve)
auc(RocCurve)


##create cumulative lift chart
plot(c(0, GainsTable$cume.pct.of.total*sum(ValData$playAction)) ~ c(0, GainsTable$cume.obs), 
     xlab = '# of cases', 
     ylab = "Cumulative", 
     type = "l")
lines(c(0, sum(ValData$playAction))~c(0, dim(ValData)[1]), 
      col="red", 
      lty=2)


##################### My computer went 32 hours and did not complete this execution
##################### Although I would like to have seen what this method would have resulted in
##################### I dont have the computational power/time to execute this
##After evaluating initial method of wanting to do Random Forest technique
##Trying boosting method
##convert playAction back to factor
ValData$playAction <- as.factor(ValData$playAction)
set.seed(27)
BoostTree <- boosting(playAction ~ ., 
                          data = TrainData, 
                          mfinal = 100)

##create confusion matrix
BoostPrediction <- predict(BoostTree, validationSet)
confusionMatrix(as.factor(BoostPrediction), 
                ValData$playAction, 
                positive = "1")
