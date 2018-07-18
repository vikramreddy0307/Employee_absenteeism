#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 00:02:37 2018

@author: vikramreddy
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression, Lars
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


dat = pd.read_excel("Absenteeism_at_work_Project.xls")
dat.info()

#missing value imputation


columns={'Disciplinary failure','Reason for absence','Month of absence','Day of the week',
         'Seasons','Transportation expense','Distance from Residence to Work','Service time',
         'Age','Hit target','Education','Son','Social drinker','Work load Average/day ',
         'Social smoker','Pet','Weight','Height','Body mass index','Absenteeism time in hours'}
for column in columns:
    dat[column].fillna(dat[column].median(),inplace=True)
    


df=dat.copy()

#recoding categorical variables
rule={0:'infectious,parasitic diseases',1:'Neoplasms',2:'Diseases of the blood',3:'Endocrine and metabolic diseases',4:'Mental and behavioural disorders', 
      5:'Diseases of the nervous system',6:'Diseases of the eye and adnexa',7:'Diseases of the ear and mastoid process',
      8:'Diseases of the circulatory system',9:'Diseases of the respiratory system',10:'Diseases of the digestive system', 
      11:'Diseases of the skin and subcutaneous tissue',12:'Diseases of the musculoskeletal system and connective tissue', 
      13:'Diseases of the genitourinary system',14:'Pregnancy, childbirth and the puerperium',15:'Certain conditions originating in the perinatal', 
      16:'Congenital malformations, deformations and chromosomal abnormalities',17:'Symptoms, signs and abnormal clinical  findings',
      18:'Injury, poisoning and certain other consequences of external causes',19:'causes of morbidity and mortality',
      21:'Factors influencing health status and contact with health services',22:'patient follow-up',23:'medical consultation',24:'blood donation',
      25:'laboratory examination',26:'unjustified absence',27:'physiotherapy',28:'dental consultation'}
rule2={0:'None',1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',
                                      6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
rule3={1:'summer',2:'autumn',3:'winter',4:'spring'}
rule4={1:'highschool',2:'graduate',3:'postgraduate',4:'master& doctrate'}
rule5={0:'No',1:'Yes'}
rule6={2:"Monday",3:"Tuesday",4:"Wednesday",5:"Thursday",6:"Friday"}
df["Reason for absence"]=df["Reason for absence"].replace(rule)
df["Month of absence"]=df["Month of absence"].replace(rule2)
df['Day of the week']=df['Day of the week'].replace(rule6)
df['Education']=df['Education'].replace(rule4)
df["Seasons"]=df["Seasons"].replace(rule3)
df["Social drinker"]=df["Social drinker"].replace(rule5)
df["Social smoker"]=df["Social smoker"].replace(rule5)
df["Disciplinary failure"]=df["Disciplinary failure"].replace(rule5)
df.isnull().any().any()




    



dat.isnull().any().any()
dat.info()

    


#heat map of numerical features
#no features are highly correlated
colormap = plt.cm.RdBu
plt.figure(figsize=(15,15))
plt.title('Pearson Correlation of Features', y=1.0, size=10)
sns.heatmap(dat[['Transportation expense','Distance from Residence to Work','Service time','Work load Average/day ',
         'Age','Hit target','Pet','Weight','Height','Body mass index','Absenteeism time in hours']].corr(),linewidths=0.2,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


#outliers plot
#there are ouliers in target variable 


plt.boxplot(dat['Transportation expense'])
plt.boxplot(dat['Distance from Residence to Work'])
plt.boxplot(dat['Service time'])
plt.boxplot(dat['Age'])
plt.boxplot(dat['Pet'])
plt.boxplot(dat['Weight'])
plt.boxplot(dat['Height'])
plt.boxplot(dat['Body mass index'])
plt.boxplot(dat['Absenteeism time in hours'])#ouliers spotted
#exploratory data analysis of factor variables  with visualisation
#boxplots of factor variables vs target variable
sns.boxplot(x='Seasons', y='Absenteeism time in hours', data=df)
sns.boxplot(x='Reason for absence', y='Absenteeism time in hours', data=df)
sns.boxplot(x='Education', y='Absenteeism time in hours', data=df)
sns.boxplot(x='Social smoker', y='Absenteeism time in hours', data=df)
sns.boxplot(x='Social drinker', y='Absenteeism time in hours', data=df)
sns.boxplot(x='Disciplinary failure', y='Absenteeism time in hours', data=df)
#histogram of factor variables 
sns.countplot(df['Reason for absence'], color='blue')
sns.countplot(df['Month of absence'], color='blue')
sns.countplot(df['Social smoker'], color='blue')
sns.countplot(df['Education'], color='blue')
sns.countplot(df['Day of the week'], color='blue')
sns.countplot(df['Social drinker'], color='blue')
sns.countplot(df['Disciplinary failure'], color='blue')



#intuition on the interactions between continuous variables
sns.pairplot(dat, vars=['Transportation expense','Distance from Residence to Work','Service time',
         'Age','Hit target','Pet','Weight','Height','Body mass index','Absenteeism time in hours'],
                 kind='reg')  
##intuition on the interactions between categorical variables
sns.pairplot(dat, vars=['Seasons','Reason for absence','Education','Social smoker','Social drinker','Disciplinary failure'],
                 kind='reg', hue='SEX')


#anova test
##analyze the differences among group means in a sample
# It simply compares the means and variations of different groups (or levels) of the categorical variable and 
#tests if there is any significant difference in their values.
#If yes, then we can say that there is an association (relationship) between the categorical predictor variable and quantitative target variable, otherwise not.
#as i am getting error due to column  names i am changing the column names
hi=dat.copy()
hi.columns=hi.columns.str.replace(' ','_')
list(hi)
hi.isnull().sum()
np.warnings.filterwarnings('ignore')
model_1=ols('Absenteeism_time_in_hours ~ C(Education)',data=hi).fit()
sm.stats.anova_lm(model_1, typ=1)
model_2=ols('Absenteeism_time_in_hours ~ C(Reason_for_absence)',data=hi).fit()
sm.stats.anova_lm(model_2, typ=1)
model_3=ols('Absenteeism_time_in_hours ~ C(Day_of_the_week)',data=hi).fit()
sm.stats.anova_lm(model_3, typ=1)
model_4=ols('Absenteeism_time_in_hours ~ C(Seasons)',data=hi).fit()
sm.stats.anova_lm(model_4, typ=1)
model_5=ols('Absenteeism_time_in_hours ~ C(Disciplinary_failure)',data=hi).fit()
sm.stats.anova_lm(model_5, typ=1)
model_6=ols('Absenteeism_time_in_hours ~ C(Social_drinker)',data=hi).fit()
sm.stats.anova_lm(model_6, typ=1)
model_7=ols('Absenteeism_time_in_hours ~ C(Social_smoker)',data=hi).fit()
sm.stats.anova_lm(model_7, typ=1)


#scaling numerical features as they have different ranges
numeric_features={'Transportation expense','Distance from Residence to Work','Service time',
         'Age','Hit target','Pet','Weight','Height','Body mass index','Work load Average/day '}
scaler= MinMaxScaler()
for column in numeric_features:
    dat[column]=scaler.fit_transform(dat[column].values.reshape(-1,1))

#one_hot_encoding for factor variables

reason=pd.get_dummies(dat['Reason for absence'])
month=pd.get_dummies(dat['Month of absence'])
education=pd.get_dummies(dat['Education'])
season=pd.get_dummies(dat['Seasons'])
day=pd.get_dummies(dat['Day of the week'])
discip=pd.get_dummies(dat['Disciplinary failure'])
soc_smok=pd.get_dummies(dat['Social smoker'])
soc_drink=pd.get_dummies(dat['Social drinker'])

#combining all the binary coded variables to original d data
encod=pd.concat([reason,month,education,season,day,discip,soc_smok,soc_drink],axis=1)




#removing all the factor variables which are used in ine hot encoding
dat=dat.drop(['Reason for absence','Month of absence','Education','Seasons',
             'Day of the week','Disciplinary failure','Social smoker','Social drinker'],axis=1)
    

#   joining encoded data to the original data
dat = dat.join(encod)



#######     creating copy of the processed data to use in PCA model    ####
pca_data=dat.copy()





###############################################################
#
#
#
######################      Feature engineering    ###################
#
#
#################################################################






target=dat['Absenteeism time in hours']
train=dat.drop(['Absenteeism time in hours'],axis=1)



#selecting top 15 features
X_new = SelectKBest(f_regression, k=40).fit_transform(train,target)
X_train, X_test, y_train, y_test = train_test_split(X_new,target, test_size=0.2,random_state=42)




#Applying after normalizing the data 
 #Applying after normalizing the data 

    
##########################
#
#
#  Removing outliers from the data
#
#
###########################    
    
    
    
 #replacing outliers less than first quantile value with 0.05 quantile value
 #replacing outliers greater than 0.95 quantile value with 0.95 quantile value

    
data_without_outliers=dat.copy()   
down=dat['Absenteeism time in hours'].quantile(0.05)  ###value=1

up=dat['Absenteeism time in hours'].quantile(0.95)    ##value =24

data_without_outliers['Absenteeism time in hours']=data_without_outliers['Absenteeism time in hours'].mask(data_without_outliers['Absenteeism time in hours'] < 1,1)
data_without_outliers['Absenteeism time in hours']=data_without_outliers['Absenteeism time in hours'].mask(data_without_outliers['Absenteeism time in hours'] >24,24)
    
target_data=data_without_outliers['Absenteeism time in hours']
train_data=data_without_outliers.drop(['Absenteeism time in hours'],axis=1)


 #selecting top 15 features
X_new_clean = SelectKBest(f_regression, k=15).fit_transform(train_data,target_data)
    
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_new_clean,target_data, test_size=0.2,random_state=42)


##################################
#
#
# creating a function that displays the results of the model
#
#
##################################



def model_results(model):
    
    
    ####################################
    #
    #
    #raw data --------   *****   with outliers    ******
    #
    #
    ####################################
    
    #fitting the raw data(with outliers tothe model)
    model.fit(X_train,y_train)
    
    #test predictions
    test_predictions=model.predict(X_test)
    
    RMSE_raw=np.sqrt(mean_squared_error(y_test, test_predictions))
    
    print("test-RMSE of data with outliers")
    print(RMSE_raw)
    print('coefficient of determination R^2 of the prediction')

    print(model.score(X_test, y_test))
    

    #####################################
    #
    #
    #clean data -------    *******    with out outliers  ******
    #
    #
    #####################################
    

   #fitting outlier removed data to the model
    model.fit(X_train_clean,y_train_clean)
    
    #test predictions
    test_pred_clean=model.predict(X_test_clean)
    
    
   #rescaling  predictions on test data as we have scaled in the beginning
    RMSE_clean=np.sqrt(mean_squared_error(y_test_clean, test_pred_clean))

    print("test-RMSE of outlier removed data(clean data) ")
    print(RMSE_clean)
    
    # Returns the coefficient of determination R^2 of the prediction.
    print('coefficient of determination R^2 of the prediction')
    print(model.score(X_test_clean, y_test_clean))
    
    return ""








############################################
#
#
#
########   Principal component analysis     ##########
#
#
#
############################################

#data preparations

pca_data['Absenteeism time in hours']=pca_data['Absenteeism time in hours'].mask(pca_data['Absenteeism time in hours'] < 1,1)
pca_data['Absenteeism time in hours']=pca_data['Absenteeism time in hours'].mask(pca_data['Absenteeism time in hours'] >24,24)
#convert it to numpy arrays
#droppinf employee ID column
data=pca_data.drop(['ID'],axis=1)
X=data.values

#target variable
target_pca=data['Absenteeism time in hours']

#passing the total number of components to the PCA    
pca = PCA(n_components=72)

#fitting the values to PCA
pca.fit(X)
    
#The amount of variance that each PC explained
var= pca.explained_variance_ratio_
    
#Cumulative Variance 
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

#graph of the variance
    
plt.plot(var1)


#############################
## from the above plot 
#The plot above shows that ~ 40 components explains around 99% variance in the data set. 
#By using PCA we have reduced 72 predictors to 40 without compromising on explained variance.
#############################  

 
#Looking at above plot I'm taking 40 variables
pca = PCA(n_components=40)

#now fitting the selected components to the data
pca.fit(X)

#PCA selected features
X1=pca.fit_transform(X)
#splitting train and test data
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X1,target_pca, test_size=0.2,random_state=42)





#################################
#
#
#creating a function that displays PCA results of their models
#
#
################################

def pca_model_results(model):
    
    
    
    #fitting training data to the model
    model.fit(X_train_pca,y_train_pca)
    #train predictions
    train_pred_pca=model.predict(X_train_pca)
    
    #RMSE of Train predictions and train data
    train_RMSE=np.sqrt(mean_squared_error(y_train_pca, train_pred_pca))
    print("RMSE on trained data of PCA model")
    print(train_RMSE)

    
    
    #test predictions
    test_pred_pca=model.predict(X_test_pca)
    
    #RMSE of test predictions and test data
    RMSE=np.sqrt(mean_squared_error(y_test_pca, test_pred_pca))
    print("test-RMSE PCA model ")
    print(RMSE)
    
    
    # Returns the coefficient of determination R^2 of the prediction.
    print('coefficient of determination R^2 of the prediction')
    print(model.score(X_test_pca, y_test_pca))
    return ""
    


#######################################
#
#
#  Building predictive models and calling the results function that i have created above
#
#
######################################    



#ridge regression
Ridge_model=Ridge()#***
model_results(Ridge_model)
pca_model_results(Ridge_model)

#random forest
rf_model=RandomForestRegressor(random_state=2)
model_results(rf_model)
pca_model_results(rf_model)


#lasso regression
lasso_model=Lasso()
model_results(lasso_model)
pca_model_results(lasso_model)

#Decision tree regressor
DT_model=DecisionTreeRegressor(random_state=3)
model_results(DT_model)
pca_model_results(DT_model)



#linear regression
lin_reg_model=LinearRegression()#****
model_results(lin_reg_model)
pca_model_results(lin_reg_model)



#lars regressor
lars_model=Lars()#****
model_results(lars_model)
pca_model_results(lars_model)





