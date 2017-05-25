# Practical Machine Learning - Predicting correctness of Weight Lifting Exercises
Michalis Frangos  
25 May 2017  






# Project Summary
This report is **just an example** on cleaning data, feature selection, model fitting, and prediction, for the purpose of completing the course on Practical Machine Learing in Coursera.

The goal of the project is to:

- use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants
- predict the manner in which they did an exercise

The report briefly discusses:

- how the model was build
- how cross validation was used
- the choices made for building the predictive model
- the expected out of sample error
- the model predictions on 20 different test cases


##Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

## Data

The training and testing  data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

Source: <http://groupware.les.inf.puc-rio.br/har>.

The variable **classe** is the one to predict.




# Cleaning data
All collected data based on which the prediction model will be built are numerical data from accelerometers etc. To clean data:

- Remove features with non-numeric data 
- Remove columns that have too many NA's
- Keep rows with complete datasets
- Remove all variables with no variation



```r
# Collect column names with non-numeric values or integers
colnames_nonumeric = c("new_window","num_window","user_name",
                       "cvtd_timestamp","raw_timestamp_part_2",
                       "raw_timestamp_part_1","X")

# Find columns with too many NA's : remove if NA ratio is >90%
percent = 90/100
na_training = sapply(training, function(x) sum(is.na(x))/nrow(training))
na_testing = sapply(testing, function(x) sum(is.na(x))/nrow(testing))
colnames_NA = union(names(training[,na_training>percent]),names(testing[,na_testing>percent]))

# Find columns with small or no variance (i.e. no usefull information)
min_var = 10 # I picked this  by looking at figure below for different values of min_var
indx_ignore = names(training) %in% c("classe",colnames_NA,colnames_nonumeric)
var_list = sapply(training[,!indx_ignore],FUN = function(x) {var(x)})
colnames_smallvar = names(var_list[var_list< min_var])

# Combine all column names to remove and remove from dataframe 
indx_ignore = names(training) %in% c(colnames_NA,colnames_nonumeric,colnames_smallvar)
training[,indx_ignore] = list(NULL)
indx_ignore = names(testing) %in% c(colnames_NA,colnames_nonumeric,colnames_smallvar)
testing[,indx_ignore] = list(NULL)

# Keep only rows with complete datasets 
training = training[complete.cases(training),]
testing  = testing[complete.cases(testing),]
```
A plot of the histograms of the features given the clean data is given below.


```r
# plot histograms of all columns
indx_classe = names(training) %in% c("classe")
df = melt(training[,!indx_classe])
ggplot(df,aes(x = value)) + facet_wrap(~variable,scales = "free") + geom_histogram(bins=50)
```

![](index_files/figure-html/plotfeatures-1.png)<!-- -->

#  Reducing the parameter space

To reduce the parameter space by keeping the dominant componenets of the parameter space I use Principal Component Analysis (PCA) given the training.

If PCA flag is 'on' apply PCA to the training and testing dataset.  


```r
flag_PCA = TRUE
if (flag_PCA){
  preProc = preProcess(training,method = 'pca',thresh=0.9)
  training.active = predict(preProc,training)
  testing.active  = predict(preProc,testing)
  
}else {
  training.active =  training
  testing.active =  testing
}
cat(" Dimension of training data :", dim(training.active))
```

```
##  Dimension of training data : 19622 15
```

# Cross validation
The caret package has the option to set the train options and do cross validation using the training set. I set the training control for cross-Validation to 5 folds for all models for computational efficiency. 

Even though not necessary, I repeat cross-validation manually based on validation data obtained from partitioning the training data.



```r
trainIndex = createDataPartition(training$classe, p = 0.7,list = FALSE)
#N = floor(nrow(training)*70/100); trainIndex = sample(1:nrow(training), N, replace=FALSE)

training = training.active[trainIndex,]
testing = training.active[-trainIndex,]

# define training control
train_control = trainControl(method="cv", number= 5, allowParallel=T)
```

# Training and validating the model


## Decision tree:



```r
# Fit model with rpart
modelFit.rpart = train(classe ~ ., data = training, trControl = train_control, method = "rpart")
pred = predict(modelFit.rpart, newdata = testing)
confMX = confusionMatrix(testing$classe,pred)
confMX
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1252   59    0  363    0
##          B  479  282    0  370    8
##          C  610   25    0  391    0
##          D  299  100    0  497   68
##          E  408  124    0  398  152
## 
## Overall Statistics
##                                           
##                Accuracy : 0.3709          
##                  95% CI : (0.3586, 0.3834)
##     No Information Rate : 0.5179          
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.183           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4108  0.47797       NA  0.24616  0.66667
## Specificity            0.8513  0.83815   0.8257  0.87920  0.83560
## Pos Pred Value         0.7479  0.24759       NA  0.51556  0.14048
## Neg Pred Value         0.5735  0.93510       NA  0.69071  0.98418
## Prevalence             0.5179  0.10025   0.0000  0.34308  0.03874
## Detection Rate         0.2127  0.04792   0.0000  0.08445  0.02583
## Detection Prevalence   0.2845  0.19354   0.1743  0.16381  0.18386
## Balanced Accuracy      0.6310  0.65806       NA  0.56268  0.75113
```

```r
accuracy.rpart = confMX$overall["Accuracy"]
fancyRpartPlot(modelFit.rpart$finalModel)
```

![](index_files/figure-html/rpart-1.png)<!-- -->


## Random forest:


```r
modelFit.rf = train(classe ~ ., data = training, trControl = train_control, method = "rf", ntree= 50)
pred = predict(modelFit.rf, newdata = testing)
confMX = confusionMatrix(testing$classe,pred)
confMX
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1639   15    8    9    3
##          B   53 1037   31    8   10
##          C    4   29  958   21   14
##          D    6    8   55  887    8
##          E    6   16   12   22 1026
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9426          
##                  95% CI : (0.9363, 0.9484)
##     No Information Rate : 0.2902          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9273          
##  Mcnemar's Test P-Value : 7.601e-07       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9596   0.9385   0.9004   0.9366   0.9670
## Specificity            0.9916   0.9787   0.9859   0.9844   0.9884
## Pos Pred Value         0.9791   0.9104   0.9337   0.9201   0.9482
## Neg Pred Value         0.9836   0.9857   0.9782   0.9878   0.9927
## Prevalence             0.2902   0.1878   0.1808   0.1609   0.1803
## Detection Rate         0.2785   0.1762   0.1628   0.1507   0.1743
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9756   0.9586   0.9431   0.9605   0.9777
```

```r
accuracy.rf = confMX$overall["Accuracy"]
plot(modelFit.rf$finalModel)
```

![](index_files/figure-html/rf-1.png)<!-- -->

# Out of sample error


```
##  Out-of-sample error for decision tree (rpart) : 0.6290569
```

```
##  Out-of-sample error for random forest  (rf) : 0.05743415
```


# Discussion 



Features data were numerical data from accelerometers etc. 

- Data were cleaned as discussed above


As this exercise is just for demontration, I tried two methods:

- Decision Tree (rpart)
- Random Forest (rf)


Cross validation:

- Based on the validation tests given this project's data, the decision tree (rpart) has a poorer performance compared to random forest (rf).

Computational complexity:

- I have set the number of trees in random forest to 50.
- I used PCA to reduce the parameter space. 
- PCA does not improve the prediction accuracy in this example; however, I used it as an example to reduce the parameter space, which can be helpful in larger datasets.

# Testing 
The prediction of the testing set via random forest: 

```
##  [1] B A A A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
