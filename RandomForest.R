

library(pacman)
p_load(tidyverse, randomForest, caret)
#caret is used to run the confusion matrix


#######################################################################################
#loading the data---- 
#######################################################################################

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

heartDisease <- read.csv(url, header = FALSE)
head(heartDisease)


#######################################################################################
#Tidying the data------
#######################################################################################

colnames(heartDisease) <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "hd") #to name the columns

str(heartDisease)

heartDisease[heartDisease == "?"] <- NA

#heartDisease[heartDisease$sex == 0, ]$sex <- "F"
#heartDisease[heartDisease$sex == 1, ]$sex <- "M"

heartDisease <- heartDisease %>% 
  mutate(sex = as.factor(ifelse(sex == 0, "F", "M")),
         hd = as.factor(ifelse(hd == 0, "Healthy", "Unhealthy")),
         cp = as.factor(cp),
         fbs = as.factor(fbs),
         restecg = as.factor(restecg),
         exang = as.factor(exang),
         slope = as.factor(slope),
         ca = as.factor(as.integer(ca)),
         thal = as.factor(as.integer(thal)))



#######################################################################################
#creating the RF model ----
#######################################################################################

set.seed(42) #because we are going to be randomly generating numbers to produce the result

heartDisease.imputed <- rfImpute(hd ~ ., data = heartDisease, iter = 6) #to impute values for the NAs in the dataset. 
#hd~. means that we want the hd column to be predicted by the data in all the other columns.
#iter specifies how many RFs rfimpute should build to estimate the missing values and 4 to 6 is most times enough.
head(heartDisease.imputed)

HDmodel <- randomForest(hd ~ .,
                        importance = TRUE,
                        data = heartDisease.imputed, 
                        Proximity = TRUE)
# proximity is set to TRUE so that we can use it to cluster samples. (google what proximity matrix is)
HDmodel # gives us an overview of the call, along with...
# 1) The OOB error rate for the forest with ntree trees. 
#    In this case ntree=500 by default
# 2) The confusion matrix for the forest with ntree trees.
#    The confusion matrix is laid out like this:
#          
#                Healthy                      Unhealthy
#          --------------------------------------------------------------
# Healthy  | Number of healthy people   | Number of healthy people      |
##         | correctly called "healthy" | incorectly called "unhealthy" |
##          | by the forest.             | by the forest                 |
#          --------------------------------------------------------------
# Unhealthy| Number of unhealthy people | Number of unhealthy peole     |
##          | incorrectly called         | correctly called "unhealthy"  |
##          | "healthy" by the forest    | by the forest                 |
#          --------------------------------------------------------------

## Now check to see if the random forest is actually big enough...
## Up to a point, the more trees in the forest, the better. You can tell when
## you've made enough when the OOB no longer improves.

#RESULT
# Call:
#   randomForest(formula = hd ~ ., data = heartDisease.imputed, Proximity = TRUE) 
# Type of random forest: classification #this wuld be regression if we were predicting an numeric value eg weight or height and it would say unsupervised if we didnt provide it with info on what it would predict.
# Number of trees: 500
# No. of variables tried at each split: 3
# 
# OOB estimate of  error rate: 16.83% #this means that 83.17% of the OOB samples were classified correctly by the RF
# Confusion matrix:
#   Healthy Unhealthy class.error
# Healthy       144        20   0.1219512
# Unhealthy      31       108   0.2230216

########################################## New info -----------------

#To plot important variables \

varImpPlot(HDmodel)

###Print out confusion matrix

y_pred <- predict(HDmodel, newdata = heartDisease.imputed[-1]) #1 is the hd variable which is being predicted and (-1) removes that variable

confusionMatrix(y_pred, heartDisease.imputed[ , 1])



#######################################################################################
#To see if the ntrees used is optimum ----
#######################################################################################

#plotting the error rates based on a matrix within the model called err.rate
HDmodel$err.rate

oob.error.data <- data.frame(
  Trees = rep(1:nrow(HDmodel$err.rate), times = 3),
  Type = rep(c("OOB", "Healthy", "Unhealthy"), each = nrow(HDmodel$err.rate)),
  Error = c(HDmodel$err.rate[ , "OOB"],
            HDmodel$err.rate[ , "Healthy"],
            HDmodel$err.rate[ , "Unhealthy"])
)

oob.error.data %>% 
  ggplot(aes(Trees, Error)) +
  geom_line(aes(color = Type))


#making model with 1000 trees

HDmodel_1000 <- randomForest(hd ~ ., 
                             data = heartDisease.imputed, 
                             ntree = 1000, 
                             Proximity = TRUE)
HDmodel_1000 #it doesn't make a difference in the model, infact it increases the error.

#to see if plotting with 1000 is better, plot error rate

oob.error.data1 <- data.frame(
  Trees = rep(1:nrow(HDmodel_1000$err.rate), times = 3),
  Type = rep(c("OOB", "Healthy", "Unhealthy"), each = nrow(HDmodel_1000$err.rate)),
  Error = c(HDmodel_1000$err.rate[ , "OOB"],
            HDmodel_1000$err.rate[ , "Healthy"],
            HDmodel_1000$err.rate[ , "Unhealthy"])
)

oob.error.data1 %>% 
  ggplot(aes(Trees, Error)) +
  geom_line(aes(color = Type))


#######################################################################################
#To see if the number of variables used at each internal node in the tree is optimum ----
#######################################################################################

oob.values <- vector(length = 10) #creating an empty vector that can hold 10 values

for (i in 1:10) { #create a loop that tests different numbers of variables at each step.
    temp.model <- temp.model <- randomForest(hd ~ ., data = heartDisease.imputed, mtry=i, ntree=1000) #building a RF using i to determine the number of variables to try at each step
    oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1] #stores the OOB error rate after running the RF that uses different values for mtry.
  }

oob.values #print out the different oob error values and the one with the lowest value is the number of variables that works best.




#######################################################################################
#Making final RF model ----
#######################################################################################


## find the minimum error
min(oob.values)
## find the optimal value for mtry...
mtryV <- which(oob.values == min(oob.values))
## create a model for proximities using the best value for mtry
model <- randomForest(hd ~ ., 
                      data = heartDisease.imputed,
                      ntree = 1000, 
                      proximity = TRUE, 
                      mtry = mtryV)
model

#######################################################################################
#Using RF model to draw MDS plot with samples ----
#######################################################################################

#this shows how each variable is related to each other

#(look up MDS plot)


distance.matrix <- as.dist(1-model$proximity) ## converting the proximity matrix into a distance matrix.

mds.plot <- cmdscale(distance.matrix, eig = TRUE, x.ret = TRUE)

## calculate the percentage of variation that each MDS axis accounts for...
mds.var.per <- round(mds.plot$eig/sum(mds.plot$eig)*100, 1)

## now make a fancy looking plot that shows the MDS axes and the variation:
mds.values <- mds.plot$points
mds.data <- data.frame(Sample=rownames(mds.values),
                       X=mds.values[,1],
                       Y=mds.values[,2],
                       Status=heartDisease.imputed$hd)

ggplot(data=mds.data, aes(x=X, y=Y, label=Sample)) + 
  geom_text(aes(color=Status)) +
  theme_bw() +
  xlab(paste("MDS1 - ", mds.var.per[1], "%", sep="")) +
  ylab(paste("MDS2 - ", mds.var.per[2], "%", sep="")) +
  ggtitle("MDS plot using (1 - Random Forest Proximities)")
ggsave(file="random_forest_mds_plot.pdf")






