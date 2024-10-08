# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics  import f1_score, accuracy_score, classification_report, confusion_matrix, recall_score, precision_score

"""# Reading the dataset"""

data=pd.read_csv('dataset.csv')
data

data.columns

data.info()

data.diagnosis.value_counts()

# Count of Diagnosis (M = Malignant, B = Benign)
plt.figure(figsize=(5,3))
sns.countplot(x='diagnosis', data=data)
plt.title('Diagnosis Count')
plt.show()

data.groupby(data.diagnosis).mean()

data.isnull().sum()

data.duplicated().sum()

data.describe()

# Pairplot for a quick overview of relationships between variables
sns.pairplot(data, hue='diagnosis', vars=['radius_mean', 'texture_mean', 'area_mean', 'perimeter_mean',"smoothness_mean"])
plt.show()

# Correlation heatmap of features
plt.figure(figsize=(20,15))
correlation = data.corr()
sns.heatmap(correlation, annot=True, fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Distribution Plots for Mean Features by Diagnosis
mean_columns = data.columns[2:12]  # Selecting columns related to mean features
plt.figure(figsize=(20,15))

for i, column in enumerate(mean_columns, 1):
    plt.subplot(3, 4, i)
    sns.histplot(data=data, x=column, hue='diagnosis', kde=True, element="step", bins=25)
    plt.title(f'Distribution of {column} by Diagnosis')

plt.tight_layout()
plt.show()

worst_features = data[['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'diagnosis']]

# Creating a pairplot for "worst" features
sns.pairplot(worst_features, hue='diagnosis', diag_kind='kde')
plt.show()

"""# Splitting the dataset"""

X=data.drop(['id','diagnosis'],axis=1)
y=data['diagnosis'].map(lambda x: 1 if x == 'M' else 0)

X

y

"""# Train Test Split"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

X_train

X_test

y_train

"""# Scaling the data"""

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train

X_train.shape

y_train.shape

y_test.shape

X_test.shape

"""# Model and Feature importance

* ## Logistic Regression
"""

LR=LogisticRegression()

LR.fit(X_train,y_train)

y_pred=LR.predict(X_test)
y_pred

print(classification_report(y_test,y_pred))

train_acc=LR.score(X_train,y_train)
test_acc=accuracy_score(y_test,y_pred)
recal=recall_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
print("Training Accuracy :", train_acc)
print("Testing Accuracy :", test_acc)
print("F1 Score :", f1)
print("Recall :", recal)
print("Precision :", prec)

conf_matrix = pd.DataFrame(data = confusion_matrix(y_test,y_pred),
                           columns = ['Predicted:0', 'Predicted:1'],
                           index =['Actual:0', 'Actual:1'])
plt.figure(figsize = (5, 3))
sns.heatmap(conf_matrix, annot = True, fmt = 'd')
plt.show()

LR.coef_

coefficients = LR.coef_[0]
importances= np.abs(coefficients)

feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances, 'Percentage': [round(val,2) for val in ((importances*100)/sum(importances))]})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
feature_importance.head()

feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(15,12))

"""> From the above barplot, we can observe the importance of each feature using Logistic Regression model.
> The top 5 important features as per the LR model are:
> * compactness_se
> * radius_se
> * texture_worst
> * concave points_worst
> * radius_worst

* ## Support Vector Machine
"""

svm=SVC(kernel='linear')

svm.fit(X_train,y_train)

y_pred=svm.predict(X_test)
y_pred

print(classification_report(y_test,y_pred))

train_acc=svm.score(X_train,y_train)
test_acc=accuracy_score(y_test,y_pred)
recal=recall_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
print("Training Accuracy :", train_acc)
print("Testing Accuracy :", test_acc)
print("F1 Score :", f1)
print("Recall :", recal)
print("Precision :", prec)

conf_matrix = pd.DataFrame(data = confusion_matrix(y_test,y_pred),
                           columns = ['Predicted:0', 'Predicted:1'],
                           index =['Actual:0', 'Actual:1'])
plt.figure(figsize = (5, 3))
sns.heatmap(conf_matrix, annot = True, fmt = 'd')
plt.show()

svm.coef_

coefficients = svm.coef_[0]
importances = np.abs(coefficients)

feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances, 'Percentage': [round(val,2) for val in ((importances*100)/sum(importances))]})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
feature_importance.head()

feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(15,12))

"""> From the above barplot, we can observe the importance of each feature using Support Vector Machine model.
> The top 5 important features as per the SVM model are:
> * compactness_se
> * fractal_dimension_worst
> * concave points_mean
> * texture_worst
> * radius_se

* ## Decision Tree
"""

DT=DecisionTreeClassifier(random_state=8)

DT.fit(X_train,y_train)

y_pred=DT.predict(X_test)
y_pred

print(classification_report(y_test,y_pred))

train_acc=DT.score(X_train,y_train)
test_acc=accuracy_score(y_test,y_pred)
recal=recall_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
print("Training Accuracy :", train_acc)
print("Testing Accuracy :", test_acc)
print("F1 Score :", f1)
print("Recall :", recal)
print("Precision :", prec)

conf_matrix = pd.DataFrame(data = confusion_matrix(y_test,y_pred),
                           columns = ['Predicted:0', 'Predicted:1'],
                           index =['Actual:0', 'Actual:1'])
plt.figure(figsize = (5, 3))
sns.heatmap(conf_matrix, annot = True, fmt = 'd')
plt.show()

DT.feature_importances_

importances=DT.feature_importances_

feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances, 'Percentage': [round(val,2) for val in ((importances*100)/sum(importances))]})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
feature_importance.head()

feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(15,12))

"""> From the above barplot, we can observe the importance of each feature using Logistic Regression model.
> The top 5 important features as per the LR model are:
> * compactness_se
> * radius_se
> * texture_worst
> * concave points_worst
> * radius_worst

* ## Random forest
"""

RF=RandomForestClassifier(n_estimators=50,random_state=1)

RF.fit(X_train,y_train)

y_pred=RF.predict(X_test)
y_pred

print(classification_report(y_test,y_pred))

train_acc=RF.score(X_train,y_train)
test_acc=accuracy_score(y_test,y_pred)
recal=recall_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
print("Training Accuracy :", train_acc)
print("Testing Accuracy :", test_acc)
print("F1 Score :", f1)
print("Recall :", recal)
print("Precision :", prec)

conf_matrix = pd.DataFrame(data = confusion_matrix(y_test,y_pred),
                           columns = ['Predicted:0', 'Predicted:1'],
                           index =['Actual:0', 'Actual:1'])
plt.figure(figsize = (5, 3))
sns.heatmap(conf_matrix, annot = True, fmt = 'd')
plt.show()

importances = RF.feature_importances_

feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances, 'Percentage': [round(val,2) for val in ((importances*100)/sum(importances))]})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
feature_importance.head()

feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(15,12))

"""> From the above barplot, we can observe the importance of each feature using Random Forest model.
> The top 5 important features as per the RF model are:
> * area_worst
> * perimeter_worst
> * concave points_worst
> * concave points_mean
> * radius_worst

* ## Gradient Boosting
"""

GB=GradientBoostingClassifier(n_estimators=12)

GB.fit(X_train,y_train)

y_pred=GB.predict(X_test)
y_pred

print(classification_report(y_test,y_pred))

train_acc=GB.score(X_train,y_train)
test_acc=accuracy_score(y_test,y_pred)
recal=recall_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
print("Training Accuracy :", train_acc)
print("Testing Accuracy :", test_acc)
print("F1 Score :", f1)
print("Recall :", recal)
print("Precision :", prec)

conf_matrix = pd.DataFrame(data = confusion_matrix(y_test,y_pred),
                           columns = ['Predicted:0', 'Predicted:1'],
                           index =['Actual:0', 'Actual:1'])
plt.figure(figsize = (5, 3))
sns.heatmap(conf_matrix, annot = True, fmt = 'd')
plt.show()

GB.feature_importances_

importances = GB.feature_importances_

feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances, 'Percentage': [round(val,2) for val in ((importances*100)/sum(importances))]})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
feature_importance.head()

feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(15,12))

"""> From the above barplot, we can observe the importance of each feature using Gradient Boosting model.
> The top 5 important features as per the GB model are:
> * concave points_worst
> * perimeter_worst
> * radius_worst
> * texture_worst
> * concave points_mean

* ## Stochastic Gradient Descent
"""

SGD=SGDClassifier(loss='modified_huber', random_state=10)

SGD.fit(X_train,y_train)

y_pred=SGD.predict(X_test)
y_pred

print(classification_report(y_test,y_pred))

train_acc=SGD.score(X_train,y_train)
test_acc=accuracy_score(y_test,y_pred)
recal=recall_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
print("Training Accuracy :", train_acc)
print("Testing Accuracy :", test_acc)
print("F1 Score :", f1)
print("Recall :", recal)
print("Precision :", prec)

conf_matrix = pd.DataFrame(data = confusion_matrix(y_test,y_pred),
                           columns = ['Predicted:0', 'Predicted:1'],
                           index =['Actual:0', 'Actual:1'])
plt.figure(figsize = (5, 3))
sns.heatmap(conf_matrix, annot = True, fmt = 'd')
plt.show()

SGD.coef_

coefficients = SGD.coef_[0]
importances = np.abs(coefficients)

feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances, 'Percentage': [round(val,2) for val in ((importances*100)/sum(importances))]})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
feature_importance.head()

feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(15,12))

"""> From the above barplot, we can observe the importance of each feature using Stochastic Gradient Descent model.
> The top 5 important features as per the SGD model are:
> * radius_se
> * concavity_mean
> * compactness_se
> * concave points_worst
> * radius_worst

* ## XGBoost
"""

XGB= XGBClassifier(random_state=0,n_estimators=88,booster='gbtree')

XGB.fit(X_train,y_train)

y_pred=XGB.predict(X_test)
y_pred

print(classification_report(y_test,y_pred))

train_acc=XGB.score(X_train,y_train)
test_acc=accuracy_score(y_test,y_pred)
recal=recall_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
print("Training Accuracy :", train_acc)
print("Testing Accuracy :", test_acc)
print("F1 Score :", f1)
print("Recall :", recal)
print("Precision :", prec)

conf_matrix = pd.DataFrame(data = confusion_matrix(y_test,y_pred),
                           columns = ['Predicted:0', 'Predicted:1'],
                           index =['Actual:0', 'Actual:1'])
plt.figure(figsize = (5, 3))
sns.heatmap(conf_matrix, annot = True, fmt = 'd')
plt.show()

XGB.feature_importances_

importances = XGB.feature_importances_

feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances, 'Percentage': [round(val,2) for val in ((importances*100)/sum(importances))]})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
feature_importance.head()

feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(15,12))

"""> From the above barplot, we can observe the importance of each feature using XGBoost model.
> The top 5 important features as per the XGB model are:
> * concave points_worst
> * concave points_mean
> * radius_worst
> * area_worst
> * texture_mean

* ## LGBM
"""

LGBM=LGBMClassifier(random_state=0,n_estimators=22)

LGBM.fit(X_train,y_train)

y_pred=LGBM.predict(X_test)
y_pred

print(classification_report(y_test,y_pred))

train_acc=LGBM.score(X_train,y_train)
test_acc=accuracy_score(y_test,y_pred)
recal=recall_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
print("Training Accuracy :", train_acc)
print("Testing Accuracy :", test_acc)
print("F1 Score :", f1)
print("Recall :", recal)
print("Precision :", prec)

conf_matrix = pd.DataFrame(data = confusion_matrix(y_test,y_pred),
                           columns = ['Predicted:0', 'Predicted:1'],
                           index =['Actual:0', 'Actual:1'])
plt.figure(figsize = (5, 3))
sns.heatmap(conf_matrix, annot = True, fmt = 'd')
plt.show()

LGBM.feature_importances_

importances = LGBM.feature_importances_

feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances, 'Percentage': [round(val,2) for val in ((importances*100)/sum(importances))]})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
feature_importance.head()

feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(15,12))

