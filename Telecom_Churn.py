# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import pandas as pd
import numpy as np
import random
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import pickle

dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn (1).csv')

dataset.head()

dataset.columns


dataset.describe() # Distribution of Numerical Variables

dataset.isna().any()

#Convert to numirac value
dataset['gender'] = dataset['gender'].str.lower().replace({'male': 1, 'female': 0})
dataset['Partner'] = dataset['Partner'].str.lower().replace({'yes': 1, 'no': 0})
dataset['Dependents'] = dataset['Dependents'].str.lower().replace({'yes': 1, 'no': 0})
dataset['PhoneService'] = dataset['PhoneService'].str.lower().replace({'yes': 1, 'no': 0})
dataset['PaperlessBilling'] = dataset['PaperlessBilling'].str.lower().replace({'yes': 1, 'no': 0})
dataset['Churn'] = dataset['Churn'].str.lower().replace({'yes': 1, 'no': 0})



#MultipleLines : Whether the customer has multiple lines or not (Yes, No, No phone service)¶ 
#look like Yes No feature but it contain 3 values. I should create new column that can 
#tell model this customer has phone service or not. but we already have 'PhoneService' columns, 
#then I decide to assume that "No phone service" has the same meaning with "No"

dataset['MultipleLines'].replace('No phone service','No', inplace=True)
dataset['MultipleLines'] = dataset['MultipleLines'].map(lambda s :1  if s =='Yes' else 0)
dataset['OnlineSecurity'].replace('no internet service','No', inplace=True)
dataset['OnlineSecurity'] = dataset['OnlineSecurity'].map(lambda s :1  if s =='Yes' else 0)
dataset['DeviceProtection'].replace('no internet service','No', inplace=True)
dataset['DeviceProtection'] = dataset['DeviceProtection'].map(lambda s :1  if s =='Yes' else 0)
dataset['TechSupport'].replace('no internet service','No', inplace=True)
dataset['TechSupport'] = dataset['TechSupport'].map(lambda s :1  if s =='Yes' else 0)
dataset['StreamingTV'].replace('no internet service','No', inplace=True)
dataset['StreamingTV'] = dataset['StreamingTV'].map(lambda s :1  if s =='Yes' else 0)
dataset['StreamingMovies'].replace('no internet service','No', inplace=True)
dataset['StreamingMovies'] = dataset['StreamingMovies'].map(lambda s :1  if s =='Yes' else 0)
dataset['OnlineBackup'].replace('no internet service','No', inplace=True)
dataset['OnlineBackup'] = dataset['OnlineBackup'].map(lambda s :1  if s =='Yes' else 0)




print(dataset['OnlineBackup'].value_counts())
print(dataset['MultipleLines'].value_counts())
print(dataset['OnlineSecurity'].value_counts())
print(dataset['DeviceProtection'].value_counts())
print(dataset['TechSupport'].value_counts())
print(dataset['StreamingTV'].value_counts())
print(dataset['StreamingMovies'].value_counts())


## because 11 rows contain " " , it means 11 missing data in our dataset
len(dataset[dataset['TotalCharges'] == " "])

## Drop missing data
dataset = dataset[dataset['TotalCharges'] != " "]





dataset.head()


sn.boxplot(x='Churn',y='tenure',data=dataset,palette='rainbow');

sn.countplot(x='tenure',data=dataset);

sn.boxplot(x='Partner',y='tenure',data=dataset,palette='rainbow');



## Pie Plots
dataset2 = dataset[['gender', 'SeniorCitizen', 'Partner',
                    'Dependents', 'tenure', 'PhoneService',
                    'MultipleLines', 'OnlineSecurity', 'DeviceProtection',
                    'TechSupport', 'StreamingTV',
                    'StreamingMovies', 'PaymentMethod', 'PaperlessBilling',
                    'Contract']]
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Pie Chart Distributions', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(6, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])
   
    values = dataset2.iloc[:, i - 1].value_counts(normalize = True).values
    index = dataset2.iloc[:, i - 1].value_counts(normalize = True).index
    plt.pie(values, labels = index, autopct='%1.1f%%')
    plt.axis('equal')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])


## Correlation Matrix
sn.set(style="white")

# Compute the correlation matrix
corr = dataset.drop(columns = ['customerID', 'Churn']).corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(110, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


## Correlation Matrix
sn.set(style="white")

# Compute the correlation matrix
corr = dataset.drop(columns = ['customerID', 'Churn' , 'Partner' , 'MonthlyCharges' , 'tenure']).corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(110, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


dataset.columns

# Removing Correlated Fields
dataset = dataset.drop(columns = ['Partner' , 'MonthlyCharges' , 'tenure'])

## Data Preparation
user_identifier = dataset['customerID']
dataset = dataset.drop(columns = ['customerID'])

dataset.info()


#Hot Encoder
from sklearn.preprocessing import LabelEncoder

def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series

dataset = dataset.apply(lambda x: object_to_int(x))
dataset.head()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns = 'Churn'), dataset['Churn'],
                                                    test_size = 0.2,
                                                    random_state = 0)

# Balancing the Training Set
y_train.value_counts()

pos_index = y_train[y_train.values == 1].index
neg_index = y_train[y_train.values == 0].index

if len(pos_index) > len(neg_index):
    higher = pos_index
    lower = neg_index
else:
    higher = neg_index
    lower = pos_index

random.seed(0)
higher = np.random.choice(higher, size=len(lower))
lower = np.asarray(lower)
new_indexes = np.concatenate((lower, higher))

X_train = X_train.loc[new_indexes,]
y_train = y_train[new_indexes]


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2


# getting the shapes
print("Shape of x_train :", X_train.shape)
print("Shape of x_test :", X_test.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of y_test :", y_test.shape)


#LogisticRegression
lr_c=LogisticRegression(random_state=0)
lr_c.fit(X_train,y_train)
lr_pred=lr_c.predict(X_test)
lr_cm=confusion_matrix(y_test,lr_pred)
lr_ac=accuracy_score(y_test, lr_pred)

#Bayes
gaussian=GaussianNB()
gaussian.fit(X_train,y_train)
bayes_pred=gaussian.predict(X_test)
bayes_cm=confusion_matrix(y_test,bayes_pred)
bayes_ac=accuracy_score(bayes_pred,y_test)

#RandomForest
rdf_c=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
rdf_c.fit(X_train,y_train)
rdf_pred=rdf_c.predict(X_test)
rdf_cm=confusion_matrix(y_test,rdf_pred)
rdf_ac=accuracy_score(rdf_pred,y_test)

# DecisionTree Classifier
dtree_c=DecisionTreeClassifier(criterion='entropy',random_state=0)
dtree_c.fit(X_train,y_train)
dtree_pred=dtree_c.predict(X_test)
dtree_cm=confusion_matrix(y_test,dtree_pred)
dtree_ac=accuracy_score(dtree_pred,y_test)

#KNN
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
knn_pred=knn.predict(X_test)
knn_cm=confusion_matrix(y_test,knn_pred)
knn_ac=accuracy_score(knn_pred,y_test)

plt.figure(figsize=(20,10))
plt.subplot(2,4,1)
plt.title("LogisticRegression_cm")
sn.heatmap(lr_cm,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.subplot(2,4,3)
plt.title("bayes_cm")
sn.heatmap(bayes_cm,annot=True,cmap="Oranges",fmt="d",cbar=False)
plt.subplot(2,4,4)
plt.title("RandomForest")
sn.heatmap(rdf_cm,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.subplot(2,4,6)
plt.title("DecisionTree_cm")
sn.heatmap(dtree_cm,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.subplot(2,4,7)
plt.title("kNN_cm")
sn.heatmap(knn_cm,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.show()


print('LogisticRegression_accuracy:\t',lr_ac)
print('RandomForest_accuracy:\t\t',rdf_ac)
print('DecisionTree_accuracy:\t\t',dtree_ac)
print('KNN_accuracy:\t\t\t',knn_ac)
print('Bayes_accuracy:\t\t\t',bayes_ac)


#Plotting the Accuracy of the models

model_accuracy = pd.Series(data=[lr_ac,bayes_ac,rdf_ac,dtree_ac,knn_ac], 
                index=['LogisticRegression','Bayes',
                                      'RandomForest','DecisionTree_Classifier','KNN'])
fig= plt.figure(figsize=(10,7))
model_accuracy.sort_values().plot.barh()
plt.title('Model Accracy')


# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, lr_pred)
accuracy_score(y_test, lr_pred)
precision_score(y_test, lr_pred) # tp / (tp + fp)
recall_score(y_test, lr_pred) # tp / (tp + fn)
f1_score(y_test, lr_pred)

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, lr_pred))


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = lr_c, X = X_train, y = y_train, cv = 10)
print("Logistic Regression Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

# Analyzing Coefficients
pd.concat([pd.DataFrame(X_train.columns, columns = ["features"]),
           pd.DataFrame(np.transpose(lr_c.coef_), columns = ["coef"])
           ],axis = 1)
    
    
    
## Feature Selection
# Recursive Feature Elimination
from sklearn.feature_selection import RFE

# Model to Test
classifier = LogisticRegression()
# Select Best X Features
rfe = RFE(lr_c, 20)
rfe = rfe.fit(X_train, y_train)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
X_train.columns[rfe.support_]


# New Correlation Matrix
sn.set(style="white")

# Compute the correlation matrix
corr = X_train[X_train.columns[rfe.support_]].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}) 


# Fitting Model to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train[X_train.columns[rfe.support_]], y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test[X_train.columns[rfe.support_]])

# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred) # tp / (tp + fp)
recall_score(y_test, y_pred) # tp / (tp + fn)
f1_score(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,
                             X = X_train[X_train.columns[rfe.support_]],
                             y = y_train, cv = 10)
print("Logistic Regression Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

# Analyzing Coefficients
pd.concat([pd.DataFrame(X_train[X_train.columns[rfe.support_]].columns, columns = ["features"]),
           pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])
           ],axis = 1)

    
    #### End of Model ####


# Formatting Final Results
final_results = pd.concat([y_test, user_identifier], axis = 1).dropna()
final_results['predicted_Churn'] = y_pred
final_results = final_results[['customerID', 'Churn', 'predicted_Churn']].reset_index(drop=True)


filename = 'classifier.pickle'
outfile = open(filename,'wb')

pickle.dump(classifier,outfile)
outfile.close()

dataset.columns
dataset['OnlineBackup'].value_counts()
dataset['Contract'].value_counts()
dataset['StreamingTV'].value_counts()
dataset['TechSupport'].value_counts()
dataset['PaymentMethod'].value_counts()





























