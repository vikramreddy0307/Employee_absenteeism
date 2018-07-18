library(readxl)
train= read_excel("Absenteeism_at_work_Project.xls")
train <- data.frame(train)
summary(train)
#missing value imputation

#install.packages('Hmisc')
library(Hmisc)

for (i in (1:21)){
  train[i]=impute(train[i],median)
}

#converting first every column to numeric
col <- c(1:21)
train[col]=as.numeric(unlist(train[col]))

#converting caegorical column to factor
for (i in c(1,2,3,4,5,12,13,15,16)) {
  train[,i]=as.factor(unlist(train[i]))
}


                                 
summary(train)
#data exploration

library(forcats)
library(dplyr)
library(ggplot2)
library(plyr)
library(DataExplorer)
library(ggthemes)
install.packages("psych")
library(psych)
pairs.panels(train)
library(grid)
library(gridExtra)
library(factoextra)
library(FactoMineR)
library(dplyr)

da=train


da$Reason.for.absence=factor(da$Reason.for.absence,levels = c(0:19,21:28),labels = c('infectious,parasitic diseases','Neoplasms','Diseases of the blood','Endocrine and metabolic diseases','Mental and behavioural disorders', 
                             'Diseases of the nervous system','Diseases of the eye and adnexa','Diseases of the ear and mastoid process',
                             'Diseases of the circulatory system','Diseases of the respiratory system','Diseases of the digestive system', 
                             'Diseases of the skin and subcutaneous tissue','Diseases of the musculoskeletal system and connective tissue', 
                             'Diseases of the genitourinary system','Pregnancy, childbirth and the puerperium','Certain conditions originating in the perinatal', 
                             'Congenital malformations, deformations and chromosomal abnormalities','Symptoms, signs and abnormal clinical  findings',
                             'Injury, poisoning and certain other consequences of external causes','causes of morbidity and mortality',
                             'Factors influencing health status and contact with health services','patient follow-up','medical consultation','blood donation',
                             'laboratory examination','unjustified absence','physiotherapy','dental consultation'))

da$Month.of.absence=factor(da$Month.of.absence,levels=c(0:12),labels=c('None','Jan','Feb','Mar','Apr','May',
                                                                         'Jun','Jul','Aug','Sep','Oct','Nov','Dec'))
da$Seasons =factor(da$Seasons,levels=c(1:4),labels=c('summer','autumn','winter','spring'))

da$Education =factor(da$Education,levels=c(1:4),labels=c('highschool','graduate','postgraduate','master& doctrate'))

da$Disciplinary.failure =factor(da$Disciplinary.failure,levels =c(0:1),labels=c('No','Yes'))
da$Social.drinker = factor(da$Social.drinker,levels =c(0:1),labels=c('No','Yes'))
da$Social.smoker =factor(da$Social.smoker,levels =c(0:1),labels=c('No','Yes'))
da$Day.of.the.week =factor(da$Day.of.the.week,levels=c(2:6),labels=c("Monday","Tuesday","Wednesday","Thursday","Friday"))


# Data exploration-

ed =ggplot(da, aes(x = Education, fill = Education)) + geom_bar()
reas = ggplot(da, aes(x = Reason.for.absence,fill=Reason.for.absence)) + geom_bar()

s.s = ggplot(da, aes(x =  Social.smoker, fill =  Social.smoker)) + geom_bar() 
s.d = ggplot(da, aes(x =  Social.drinker, fill =  Social.drinker)) + geom_bar() 
disc = ggplot(da, aes(x =  Disciplinary.failure, fill =  Disciplinary.failure)) + geom_bar() 
month= ggplot(da, aes(x =  Month.of.absence, fill =  Month.of.absence)) + geom_bar() 

day = ggplot(da, aes(x = Day.of.the.week, fill =  Day.of.the.week)) + geom_bar() 
Seas= ggplot(da, aes(x =   Seasons,fill = Seasons)) + geom_bar()

plot(ed)
plot(reas)
plot(s.s)
plot(s.d)
plot(disc)
plot(Day)
plot(month)
plot(seas)
#loadind dplyr after loading plyr is causing issues

detach("package:plyr", unload=TRUE) 

#selecting samples which have absenteeism time hours > 0

absent=as.data.frame( da %>% select(everything()) %>% filter(Absenteeism.time.in.hours > 0))

#percentage absenteeism hours(>0) vs  seasons

season = as.data.frame(absent %>% group_by(Seasons) %>% summarise(count= n(), percent = round(count*100/nrow(absent),1))%>% arrange(desc(count)))
ggplot(season,aes(x= reorder(Seasons,percent), y= percent, fill = Seasons)) + geom_bar(stat='identity') + coord_flip() +
  geom_text(aes(label = percent), vjust = 1.1, hjust = 1.2) + xlab('Seasons')

#percentage absenteeism hours vs  disciplinary failure
disciplinary =as.data.frame(absent %>% group_by(Disciplinary.failure) %>% summarise(count= n(), percent = round(count*100/nrow(absent),1))%>% arrange(desc(count)))
ggplot(disciplinary,aes(x= reorder(Disciplinary.failure,percent), y= percent, fill = Disciplinary.failure)) + geom_bar(stat='identity') + coord_flip() +
  geom_text(aes(label = percent), vjust = 1.1, hjust = 1.2) + xlab('Disciplinary failure')

#percentage absenteeism hours vs Reason for abscence

Reason =as.data.frame(absent %>% group_by(Reason.for.absence) %>% summarise(count= n(), percent = round(count*100/nrow(absent),1))%>% arrange(desc(count)))
ggplot(Reason,aes(x = reorder(Reason.for.absence,percent), y= percent, fill= Reason.for.absence)) + geom_bar(stat = 'identity') + coord_flip() + theme(legend.position='none') +  
  geom_text(aes(label = percent), vjust = 0.5, hjust = 1.1) + xlab('Reason for absence')
##percentage absenteeism hours vs  day of the week
dayofweek =as.data.frame(absent %>% group_by(Day.of.the.week) %>% summarise(count= n(), percent = round(count*100/nrow(absent),1))%>% arrange(desc(count)))
ggplot(dayofweek,aes(x = reorder(Day.of.the.week,percent), y= percent, fill= Day.of.the.week)) + geom_bar(stat = 'identity') + coord_flip() + theme(legend.position='none') +  
  geom_text(aes(label = percent), vjust = 0.5, hjust = 1.1) + xlab('Day.of.the.week')
##percentage absenteeism hours vs  social smoker
socialsmoker =as.data.frame(absent %>% group_by(Social.smoker) %>% summarise(count= n(), percent = round(count*100/nrow(absent),1))%>% arrange(desc(count)))
ggplot(socialsmoker,aes(x = Social.smoker, y= percent, fill= Social.smoker)) + geom_bar(stat = 'identity') + coord_flip() + theme(legend.position='none') +  
  geom_text(aes(label = percent), vjust = 0.5, hjust = 1.1) + xlab('Social.smoker')
##percentage absenteeism hours vs social drinker

socialdrinker =as.data.frame(absent %>% group_by(Social.drinker) %>% summarise(count= n(), percent = round(count*100/nrow(absent),1))%>% arrange(desc(count)))
ggplot(socialdrinker,aes(x = Social.drinker, y= percent, fill= Social.drinker)) + geom_bar(stat = 'identity') + coord_flip() + theme(legend.position='none') +  
  geom_text(aes(label = percent), vjust = 0.5, hjust = 1.1) + xlab('Social.drinker')

#percentage of absent people vs education
education=as.data.frame(absent %>% group_by(Education) %>% summarise(count= n(), percent = round(count*100/nrow(absent),1))%>% arrange(desc(count)))
ggplot(education,aes(x = Education, y= percent, fill= Education)) + geom_bar(stat = 'identity') + coord_flip() + theme(legend.position='none') +  
  geom_text(aes(label = percent), vjust = 0.5, hjust = 1.1) + xlab('Education')
#percentage of absent people vs people having pet
pet=as.data.frame(absent %>% group_by(Pet) %>% summarise(count= n(), percent = round(count*100/nrow(absent),1))%>% arrange(desc(count)))
ggplot(pet,aes(x = Pet, y= percent, fill= Pet)) + geom_bar(stat = 'identity') + coord_flip() + theme(legend.position='none')   +
  geom_text(aes(label = percent), vjust = 0.5, hjust = 1.1) + xlab('pet')
#percentage of absent people vs age
age=as.data.frame(absent %>% group_by(Age) %>% summarise(count= n(), percent = round(count*100/nrow(absent),1))%>% arrange(desc(count)))
ggplot(age,aes(x = Age, y= percent, fill= Age)) + geom_bar(stat = 'identity') + coord_flip() + theme(legend.position='none') +  
  geom_text(aes(label = percent), vjust = 0.5, hjust = 1.1) + xlab('age')

#percentage of absent people vs weight
weight=as.data.frame(absent %>% group_by(Weight) %>% summarise(count= n(), percent = round(count*100/nrow(absent),1))%>% arrange(desc(count)))
ggplot(weight,aes(x = Weight, y= percent, fill= Weight)) + geom_bar(stat = 'identity') + coord_flip() + theme(legend.position='none') +  
  geom_text(aes(label = percent), vjust = 0.5, hjust = 1.1) + xlab('weight')

#percentage of absent people vs dist
dist=as.data.frame(absent %>% group_by(Distance.from.Residence.to.Work) %>% summarise(count= n(), percent = round(count*100/nrow(absent),1))%>% arrange(desc(count)))
ggplot(dist,aes(x = Distance.from.Residence.to.Work, y= percent, fill= Distance.from.Residence.to.Work)) + geom_bar(stat = 'identity') + coord_flip() + theme(legend.position='none') +  
  geom_text(aes(label = percent), vjust = 0.5, hjust = 1.1) + xlab('dist from residance')

#percentage of absent people vs workload
workload=as.data.frame(absent %>% group_by(Work.load.Average.day) %>% summarise(count= n(), percent = round(count*100/nrow(absent),1))%>% arrange(desc(count)))
ggplot(workload,aes(x =Work.load.Average.day, y= percent, fill= Work.load.Average.day)) + geom_bar(stat = 'identity') + coord_flip() + theme(legend.position='none') +  
  geom_text(aes(label = percent), vjust = 0.5, hjust = 1.1) + xlab('work load avg')


#histogram of numerical features
hist(train$Transportation.expense)
hist(train$Distance.from.Residence.to.Work)
hist(train$Service.time)
hist(train$Age)
hist(train$Work.load.Average.day)
hist(train$Hit.target)
hist(train$Pet)
hist(train$Weight)
hist(train$Height)
hist(train$Son)
hist(train$Body.mass.index)
hist(train$Absenteeism.time.in.hours)
#boxplots of numerical features
boxplot(train$Transportation.expense)
boxplot(train$Distance.from.Residence.to.Work)
boxplot(train$Service.time)
boxplot(train$Age)
boxplot(train$Work.load.Average.day)
boxplot( train$Hit.target)
boxplot(train$Weight)
boxplot(train$Body.mass.index)
boxplot(train$Absenteeism.time.in.hours)
#anova test
#analyze the differences among group means in a sample
# It simply compares the means and variations of different groups (or levels) of the categorical variable and 
#tests if there is any significant difference in their values.
#If yes, then we can say that there is an association (relationship) between the categorical predictor variable and quantitative target variable, otherwise not.
library(RcmdrMisc)
#anova by month of abscence
AnovaModel_season =(lm(Absenteeism.time.in.hours ~ Seasons, data = train))
Anova(AnovaModel_season)

# Absence Rate By Pet

AnovaModel_reason=(lm(da$Absenteeism.time.in.hours ~ Reason.for.absence, data = da))
Anova(AnovaModel_reason)

# By Season

AnovaModel_month = (lm(da$Absenteeism.time.in.hours ~ Month.of.absence, data = da))
Anova(AnovaModel_month)

# By Social smoker

AnovaModel_smoker = (lm(da$Absenteeism.time.in.hours ~ Social.smoker, data = da))
Anova(AnovaModel_smoker)

# By Social drinker
AnovaModel_drinker = (lm(da$Absenteeism.time.in.hours ~ Social.drinker, data = da))
Anova(AnovaModel_drinker) 

# By Education

AnovaModel_education = (lm(da$Absenteeism.time.in.hours ~ Education, data = da))
Anova(AnovaModel_education)

# By Disciplanary failure
AnovaModel_discipline = (lm(da$Absenteeism.time.in.hours ~ Disciplinary.failure, data = da))
Anova(AnovaModel_discipline)

# min max scaling because data has differnt ranges
# not scaling target variable
for (i in c(1:13,15:16,18:20)){
  if (class(train[,i])=="numeric"){
    for (j in c(1:740)){ 
      train[j,i]=(train[j,i]-min(train[i]))/(max(train[i])-min(train[i]))
    }
  }
}
#replacing outliers
##replacing values greater than 0.95 quantile with values present in 0.95 quantile
#replacing values less than 0.05 quantile with values present in 0.05 quantile

qn = quantile(train$Absenteeism.time.in.hours, c(0.05, 0.95), na.rm = TRUE)
print(qn)
train$Absenteeism.time.in.hours[train$Absenteeism.time.in.hours<qn[1]]=qn[1]
train$Absenteeism.time.in.hours[train$Absenteeism.time.in.hours>qn[2]]=qn[2]

#one hot encoding of categorical variables
library(dummies)
hi=data.frame(dummy(train$Reason.for.absence))
train$Reason.for.absence=NULL
hi1=data.frame(dummy(train$Month.of.absence))
train$Month.of.absence=NULL
hi2=data.frame(dummy(train$Day.of.the.week))
train$Day.of.the.week=NULL
hi3=data.frame(dummy(train$Education))
train$Education=NULL
hi4=data.frame(dummy(train$Social.smoker))
train$Social.smoker=NULL
hi5=data.frame(dummy(train$Social.drinker))
train$Social.drinker=NULL
hi6=data.frame(dummy(train$Disciplinary.failure))
train$Disciplinary.failure=NULL
hi7=data.frame(dummy(train$Seasons))
train$Seasons=NULL

#combining binary features with original data

train=cbind(train,hi,hi1,hi2,hi3,hi4)
train=cbind(train,hi5,hi6,hi7)

#correlation plot 
library(caret)
library(corrplot)
target=train["Absenteeism.time.in.hours"]
corm=train[-1]#removing ID column

#coorelation matrix
matrix=cor(corm)
#correlation plot
corrplot(matrix, method="pie")
#removing features with value greater than 0.95
hc = findCorrelation(matrix, cutoff=0.95) #  we can putt any value as a "cutoff" 
#sorting out the columns to be removes
hc = sort(hc)
#removing highly correlated columns
reduced_Data = corm[,-c(hc)]
#combining the clean data with the factor variables of original data
new_train=cbind(train,reduced_Data)
#test data removing ID column
test=new_train[550:740,-1]
#install.packages('DAAG')
library(DAAG)
#feature selection using boruta package
library(Boruta)
#new_train$Absenteeism.time.in.hours=log(new_train$Absenteeism.time.in.hours)
finail.boruta=Boruta(Absenteeism.time.in.hours~., data = new_train[,-1], doTrace = 2)
selected_features=getSelectedAttributes(finail.boruta, withTentative = F)
set.seed(123)
#creating formula from boruta selected features
formula=as.formula(paste("Absenteeism.time.in.hours~",paste(selected_features,collapse = "+")))


#predictive models

# linear model
train_control = trainControl(method = "repeatedcv", 
                        number = 10)
options(warn=-1)   
lm_model= train(formula,data=new_train[1:500,-1],
      metric="RMSE", method="lm",trControl=train_control)
#predictions
lm_pred=predict(lm_model,test)
#RMSE
print(RMSE(lm_pred,test$Absenteeism.time.in.hours))

# estimate variable importance
lm_importance <- varImp(lm_model, scale=FALSE)
# summarize model
print(lm_model)
# plot importance
plot(lm_importance)



#random forest model
rf_model= train(formula,data=new_train[1:550,-1],
              metric="RMSE", method="rf",trControl=train_control)
#predictions
rf_pred=predict(ridge_model,test)
#RMSE
print(RMSE(rf_pred,test$Absenteeism.time.in.hours))


# summarize model
print(rf_model)
# plot model
plot(rf_model)


#decision tree model
dt_model=train(formula,data=new_train[1:550,-1],
                 metric="RMSE", method="rpart",trControl=train_control)
#predictions
dt_pred=predict(dt_model,test)
#RMSE
print(RMSE(dt_pred,test$Absenteeism.time.in.hours))


# summarize model
print(dt_model)
# plot model
plot(dt_model)



###################
#
#
##removing certain features from the  model as heir p_value is very high not statistically significant
#
#
#
#
#
###################

new=c('Reason.for.absence.1','Reason.for.absence.7','Reason.for.absence.9',
      'Reason.for.absence.10','Reason.for.absence.11',
  'Reason.for.absence.12','Reason.for.absence.13','Reason.for.absence.14','Reason.for.absence.18',
  'Reason.for.absence.19','Reason.for.absence.21','Reason.for.absence.22','Reason.for.absence.23',
  'Reason.for.absence.26','Reason.for.absence.28')
#creating formula from boruta selected features
new_formula=as.formula(paste("Absenteeism.time.in.hours~",paste(new,collapse = "+")))
# linear model
train_control =trainControl(method = "repeatedcv", 
                              number = 10, 
                              repeats = 6)
new_lm_model= train(new_formula,data=new_train,
              metric="RMSE", method="lm",trControl=train_control)
print(new_lm_model)
print(summary(new_lm_model))
#predictions
new_lm_pred=predict(new_lm_model,test)
#RMSE
print(RMSE(new_lm_pred,test$Absenteeism.time.in.hours))

# estimate variable importance
new_lm_importance <- varImp(new_lm_model, scale=FALSE)
# summarize importance
print(new_lm_model)
#summary of the model
print(summary(new_lm_model))
# plot importance
plot(new_lm_importance)


#ridge model
new_ridge_model= train(new_formula,data=new_train,
              metric="RMSE", method="ridge",trControl=train_control)
#print model
print(new_ridge_model)
#summary of the model
print(summary(new_ridge_model))
#predictions
new_ridge_pred=predict(new_ridge_model,test)
#RMSE
print(RMSE(new_ridge_pred,test$Absenteeism.time.in.hours))

# losses every month can we project in 2011 if same trend of absenteeism continues
 ggplot(da, aes(x =  Month.of.absence), fill =  new_ridge_model) + geom_bar(fill='red') 

# estimate variable importance
new_ridge_importance = varImp(new_ridge_model, scale=FALSE)
# summarize importance
print(new_ridge_importance)
# plot importance
plot(new_ridge_importance)


#random forest model
new_rf_model= train(new_formula,data=new_train[1:550,-1],
                 metric="RMSE", method="rf",trControl=train_control)

pred_new_rf_model=predict(new_rf_model,test)
print(RMSE(new_rf_pred,test$Absenteeism.time.in.hours))


# summarize importance
print(new_rf_model)
# plot importance
plot(new_rf_model)


#decision tree model
new_dt_model= train(new_formula,data=new_train[1:550,-1],
                 metric="RMSE", method="rpart",trControl=train_control)
#predictions
new_dt_pred=predict(new_dt_model,test)
#RMSE
print(RMSE(new_dtpred,test$Absenteeism.time.in.hours))



# summarize importance
print(new_dt_model)
# plot importance
plot(new_dt_model)


############################
# from the above models even after removing statistically unsignificant variables
#the RMSE and R-squared value is very low
############################



####################
#
#
#Principal component analysis
#
#
####################



#divide the new data
pca.train = train[1:550,-1]
pca.test =train[551:740,-1]
 #principal component analysis
prin_comp <- prcomp(pca.train)
 #outputs the mean of variables
prin_comp$center
 
 #outputs the standard deviation of variables
prin_comp$scale
dim(prin_comp$x)
biplot(prin_comp, scale = 0)
 
 #compute standard deviation of each principal component
std_dev = prin_comp$sdev
 
 #compute variance
pr_var = std_dev^2
 #proportion of variance explained
prop_varex =pr_var/sum(pr_var)
 #scree plot
plot(prop_varex, xlab = "Principal Component",
        ylab = "Proportion of Variance Explained",
        type = "b")
 
 #cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
        ylab = "Cumulative Proportion of Variance Explained",
        type = "b")
 #add a training set with principal components
train.data = data.frame(Absenteeism.time.in.hours = pca.train$Absenteeism.time.in.hours, prin_comp$x)
 
 #we are interested in first 40 PCAs as we have seen from the graph
  # and the target variable ,so in total 41(including target variable)
 train.data =train.data[,1:41]
 
 #transform test into PCA
test.data=predict(prin_comp, newdata = pca.test)
test.data= as.data.frame(test.data)
 
 #select the first 40 components
test.data=test.data[,1:40]
 #linear regression
set.seed(123)
train_control=trainControl(method = "repeatedcv", 
                               number = 10, 
                               repeats = 6)
pca_lm_model= train(Absenteeism.time.in.hours~.,data=train.data,
                      metric="RMSE", method="lm",trControl=train_control)
 
print(pca_lm_model) 
print(summary(pca_lm_model))
 #make prediction on test data
pca.lm.prediction = predict(pca_lm_model, test.data)
 # absence in every month in 2011 if same trend of absenteeism continues
pca.lm.trend=data.frame(da[551:740,-1]$Month.of.absence,pca.lm.prediction)
ggplot(pca.lm.trend, aes(x = da.551.740...1..Month.of.absence , y = pca.lm.prediction) ) + geom_bar(stat = 'identity',fill='blue') 
 
 #finding RMSE on test data
print(RMSE(pca.lm.prediction,pca.test$Absenteeism.time.in.hours))
 
 
 
 
 #Ridge regression
set.seed(123)
train_control = trainControl(method = "repeatedcv", 
                               number = 10, 
                               repeats = 6)
pca_ridge_model= train(Absenteeism.time.in.hours~.,data=train.data,
                      metric="RMSE", method="ridge",trControl=train_control)
 
print(pca_ridge_model) 
print(summary(pca_ridge_model))
 #make prediction on test data
pca.ridge.prediction = predict(pca_ridge_model, test.data)
 # absence  in every month in 2011 if same trend of absenteeism continues
pca.ridge.trend=data.frame(da[551:740,-1]$Month.of.absence,pca.ridge.prediction)
ggplot(pca.ridge.trend, aes(x = da.551.740...1..Month.of.absence , y = pca.ridge.prediction) ) + geom_bar(stat = 'identity') 
 
 #finding RMSE on test data
print(RMSE(pca.ridge.prediction,pca.test$Absenteeism.time.in.hours))
 
 
 #Random forest
set.seed(123)
train_control = trainControl(method = "repeatedcv", 
                               number = 10, 
                               repeats = 6)
pca_rf_model=train(Absenteeism.time.in.hours~.,data=train.data,
                         metric="RMSE", method="rf",trControl=train_control)
print(pca_rf_model) 
print(summary(pca_rf_model))
 
 #make prediction on test data
pca.rf.prediction = predict(pca_rf_model, test.data)
 # absence in every month in 2011 if same trend of absenteeism continues
pca.rf.trend=data.frame(da[551:740,-1]$Month.of.absence,pca.rf.prediction)
ggplot(pca.rf.trend, aes(x = da.551.740...1..Month.of.absence , y = pca.rf.prediction) ) + geom_bar(stat = 'identity') 
 
 #finding RMSE on test data
print(RMSE(pca.rf.prediction,pca.test$Absenteeism.time.in.hours))
 
 
 #Decision tree
set.seed(123)
train_control = trainControl(method = "repeatedcv", 
                              number = 10, 
                              repeats = 6)
 
pca_dt_model=train(Absenteeism.time.in.hours~.,data=train.data,
                    metric="RMSE", method="rpart",tuneLength = 10, trControl=train_control)
print(pca_dt_model) 
 print(summary(pca_dt_model))
 
 #make prediction on test data
pca.dt.prediction = predict(pca_dt_model, test.data)
# absence  in every month in 2011 if same trend of absenteeism continues
pca.dt.trend=data.frame(da[551:740,-1]$Month.of.absence,pca.dt.prediction)
ggplot(pca.dt.trend, aes(x = da.551.740...1..Month.of.absence , y = pca.dt.prediction) ) + geom_bar(stat = 'identity') 
 
#finding RMSE on test data
print(RMSE(pca.dt.prediction,pca.test$Absenteeism.time.in.hours))
 
 
 #lasso regression
set.seed(123)
train_control = trainControl(method = "repeatedcv", 
                              number = 10, 
                              repeats = 6)
pca_lasso_model=train(Absenteeism.time.in.hours~.,data=train.data,
                    metric="RMSE", method="lasso",tuneLength = 10,trControl=train_control)
print(pca_lasso_model) 
print(summary(pca_lasso_model))
 
 #make prediction on test data
pca.lasso.prediction = predict(pca_lasso_model, test.data)
 # absence in every month in 2011 if same trend of absenteeism continues
pca.lasso.trend=data.frame(da[551:740,-1]$Month.of.absence,pca.lasso.prediction)
ggplot(pca.lasso.trend, aes(x = da.551.740...1..Month.of.absence , y = pca.lasso.prediction) ) + geom_bar(stat = 'identity') 
 
 #finding RMSE on test data
print(RMSE(pca.lasso.prediction,pca.test$Absenteeism.time.in.hours))
 
 