# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
# %matplotlib inline


#Importing the dataset using pandas
rain = pd.read_csv("weatherAUS.csv")

#chechking if there is any categorical features and store in the variable categorical_features
categorical_features = [column_name for column_name in rain.columns if rain[column_name].dtype == 'O']

# taking all the numerical features and store it in the variable numerical_features
numerical_features = [column_name for column_name in rain.columns if rain[column_name].dtype != 'O']

# replacing categorical_features with unique values
for each_feature in categorical_features:
   unique_values = len(rain[each_feature].unique())
   print("Cardinality(no. of unique values) of {} are: {}".format(each_feature, unique_values))

rain['Date'] = pd.to_datetime(rain['Date'])
rain['year'] = rain['Date'].dt.year
rain['month'] = rain['Date'].dt.month
rain['day'] = rain['Date'].dt.day

#droping date as it has no part to play in solving the weather prediction
rain.drop('Date', axis = 1, inplace = True)

#checking if there is any categorical_features are with null values NA
categorical_features = [column_name for column_name in rain.columns if rain[column_name].dtype == 'O']
rain[categorical_features].isnull().sum()

#if there is any cate are with null values they are being replaced with unique values
categorical_features_with_null = [feature for feature in categorical_features if rain[feature].isnull().sum()]
for each_feature in categorical_features_with_null:

    mode_val = rain[each_feature].mode()[0]
    rain[each_feature].fillna(mode_val,inplace=True)

numerical_features = [column_name for column_name in rain.columns if rain[column_name].dtype != 'O']
rain[numerical_features].isnull().sum()


#taking the features with outliers and normalizing the features with upperlimit and lower limit
features_with_outliers = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'WindGustSpeed','WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
for feature in features_with_outliers:
    q1 = rain[feature].quantile(0.25)
    q3 = rain[feature].quantile(0.75)
    IQR = q3-q1
    lower_limit = q1 - (IQR*1.5)
    upper_limit = q3 + (IQR*1.5)
    rain.loc[rain[feature]<lower_limit,feature] = lower_limit
    rain.loc[rain[feature]>upper_limit,feature] = upper_limit

# checking hte same null value NA for numerical features and replaving the with unique values
numerical_features_with_null = [feature for feature in numerical_features if rain[feature].isnull().sum()]
for feature in numerical_features_with_null:
    mean_value = rain[feature].mean()
    rain[feature].fillna(mean_value,inplace=True)

#this function replaces the categorial variable YES and NO to unique numerical variables.
def encode_data(feature_name):

    ''' 

    This function takes feature name as a parameter and returns mapping dictionary to replace(or map) categorical data with numerical data.

    '''

    mapping_dict = {}

    unique_values = list(rain[feature_name].unique())

    for idx in range(len(unique_values)):

        mapping_dict[unique_values[idx]] = idx

    return mapping_dict

rain['RainToday'].replace({'No':0, 'Yes': 1}, inplace = True)

rain['RainTomorrow'].replace({'No':0, 'Yes': 1}, inplace = True)

rain['WindGustDir'].replace(encode_data('WindGustDir'),inplace = True)

rain['WindDir9am'].replace(encode_data('WindDir9am'),inplace = True)

rain['WindDir3pm'].replace(encode_data('WindDir3pm'),inplace = True)

rain['Location'].replace(encode_data('Location'), inplace = True)


#Seperating the Dependent variable from the independent variables
X = rain.drop(['RainTomorrow'],axis=1)
y = rain['RainTomorrow']

#Spliting the data in random form for training and testing using sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

print("Length of Training Data: {}".format(len(X_train)))
print("Length of Testing Data: {}".format(len(X_test)))


# code beginning of logistic regression
from sklearn.linear_model import LogisticRegression
# using LogisticRegression method from sklearn module with liblinear as solver and no random state
classifier_logreg = LogisticRegression(solver='liblinear', random_state=0)
#training the algorithm with the training data created
classifier_logreg.fit(X_train, y_train)

y_pred = classifier_logreg.predict(X_test)

#generating accuracy_score
from sklearn.metrics import accuracy_score
LR_ac = accuracy_score(y_test,y_pred)
print("Accuracy Score Logistic Regression : {}".format(LR_ac))

# Code begning of KNN
from sklearn.neighbors import NearestNeighbors
#number of neighbours are being two since this is a teo class problem
neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train,y_train)
KN_ac = neigh.score(X_test, y_test, sample_weight=None)
print("Accuracy Score of KNN : {}".format(KN_ac))



# code beginning of Decission Tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
ac_DT = clf.score(X_test,y_test)
print("Accuracy Score Decision Tree : {}".format(ac_DT))


#Code begining of SVM
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)
ac_SVM = clf.score(X_test,y_test, sample_weight=None)
print("Accuracy Score SVM : {}".format(ac_SVM))
