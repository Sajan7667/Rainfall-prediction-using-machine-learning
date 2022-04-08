# Rainfall-prediction-using-machine-learning
Rainfall prediction is crucial for increasing agricultural productivity which in turn secures food and quality water supply for citizens of one's country.

The weatherAUS dataset is a data frame containing over 140,000 daily observations from over 45 Australian weather stations. The date of observation (a Date object). The common name of the location of the weather station. The minimum temperature in degrees celsius

The data for this article can be found [here](https://www.kaggle.com/datasets/gauravduttakiit/weather-in-aus)

## Objective 
1. Data understanding and exploring

3. Data cleaning

    • Handling missing values
   
   • Outliers treatment
   
3. Exploratory data analysis
   
4. Split the data into train and test set
    
    • Scale the data (normalization)
    
5. Model building
    
    Train the model with various algorithm such as Logistic regression, Decision Tree, KNN.
    
6. Model evaluation
     
     • As we see that the data is heavily imbalanced, Accuracy may not be the correct measure for this particular case
     
     • We have to look for a balance between Precision and Recall over Accuracy
     
     • We also have to find out the good ROC score with high TPR and low FPR in order to get the lower number of misclassifications.
     
## Software Used

	Python Language
      
  •	Models

	 Decision Tree Classification 

	 Logistic Regression 

	 KNN

 •	Pandas

## Decision Tree Classification 
  Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value
of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.
![image](https://user-images.githubusercontent.com/88305984/160349975-bf1eb2bb-5829-4ca5-aa7c-97786b6c529e.png)

 

## Logistic Regression

Logistic Regression is one of the most used ML algorithms in binary classification. Logistic Regression was used in the biological sciences in early 20th century.
It was then used in many social science applications. Logistic Regression is used when the dependent variable(target) is categorical.
![image](https://user-images.githubusercontent.com/88305984/160349928-2af28e1c-2cda-44bc-9272-cfbe927da04f.png)

## KNN
This algorithm is used to solve the classification model problems. K-nearest neighbor or K-NN algorithm basically creates an imaginary boundary to classify the data.
When new data points come in, the algorithm will try to predict that to the nearest of the boundary line.

Therefore, larger k value means smother curves of separation resulting in less complex models. 
Whereas, smaller k value tends to overfit the data and resultingin complex models.
![image](https://user-images.githubusercontent.com/88305984/162375254-5b914309-dfc5-4942-b913-9ffc7815313f.png)
