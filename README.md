# DS_Project_1-AnomaData-Automated-Anomaly-Detection-for-Predictive-Maintenance-
A. #Importing data and storing it in variable called 'data'

B. # Converting the excel data into the dataframe
   # Why ?
   #1. DataFrames are designed for easy handling of structured data
   (rows and columns), Handling missing data, Sorting data etc.
   #2. Pandas provides a wide range of built-in functions that are
   optimized for DataFrame objects.
C. Exploratory Data Analysis
   #Analyzing Rows and Columns df.shape
   (18398, 62)
   #Information about non-null values 

D. #Counting the no. of Null values for each column

E. #Finding Duplicate values

F. #Counting Unique value in each column
   ### Seperating Input and Output columns
   ##Zeroes and ones count in column 'y'

G. #Representing Zeroes and ones count in bar chart
   #Representing Zeroes and ones count in pie plot

H. # Now Install library which will help to balance the dataset 

I. """ We will use Oversampling technique - SMOTE, to balance the dataset
   which will scale up the minority classes to match up with the majority
   class """

J. Data Splitting
   # Splitting the data into training and testing dataset from
   sklearn.model_selection import train_test_split X_train, X_test, Y_train,

K. FEATURE ENGINEERING
   #Will convert raw data to useful features for ML model #success of a ML
   model largely depends on the quality of the features used in the model.
   # Extracting all numeric features in one variable

L. # Plot the histograms
   #Observing correlation b/w all features

M. #Scaling the columns in order to have more uniform distribution - We apply
   Yeo-Johnson Transform
   ## To display X_scaled as a DataFrame.
   ## Now, further splitting the data into training and testing based on
      scaled values.

N. #Importing PCA library #Telling PCA to retain 90% of useful features and then
   create new dimensions
   # Each value will tell how much % of usefulness it contributes to the
    entire dataset

O. #To display how many dimensions we will feed now in our model
   # Will now again do train test split but now with the X_pca
 
P. #Since it is a classification problem, We will use the following
   models for it and will compare their accuracy on test data against
   each other.

   #1. Logistic Regression from sklearn.linear_model import
LogisticRegression model1 = LogisticRegression()
model1.fit(X_train_sc_pca,Y_train_sc_pca)
model1.score(X_test_sc_pca,Y_test_sc_pca)
0.8383036935704514

#2. Decision Tree
from sklearn import tree
model2 = tree.DecisionTreeClassifier()
model2.fit(X_train_sc_pca,Y_train_sc_pca)
model2.score(X_test_sc_pca,Y_test_sc_pca)
0.9800273597811218

#3. KNN
from sklearn.neighbors import KNeighborsClassifier
model3 = KNeighborsClassifier(n_neighbors=3)
model3.fit(X_train_sc_pca,Y_train_sc_pca)
model3.score(X_test_sc_pca,Y_test_sc_pca)
0.9941176470588236

#4. Random Forest
from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier(n_estimators=30)
model4.fit(X_train_sc_pca,Y_train_sc_pca)
model4.score(X_test_sc_pca,Y_test_sc_pca)
0.9991792065663475

Conclusion
Considering the Classification report from the above four models, we found Random Forest
Algorithm - accuracy, precision, f1-score is acheieving 99.99 % accuracy on test data set. With a
dataset comprising over 18,000 rows and utilizing binary labels for anomaly identification, the
model has demonstrated exceptional performance, achieving an outstanding accuracy of
99.99%. This high level of accuracy underscores the model's reliability in detecting anomalies
and predicting machine breakdowns, which can significantly reduce downtime, minimize risks,
and optimize maintenance schedules across industries.




   
