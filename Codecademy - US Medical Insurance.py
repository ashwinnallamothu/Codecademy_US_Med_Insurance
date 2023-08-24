#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv

# Open the insurance.csv file in read mode as alias using 'open' function.
with open('/Users/ashwinnallamothu/Desktop/Medical_Insurance/insurance.csv', 'r') as insurance_data:
  # Create a CSV reader object
  insurance_reader = csv.DictReader(insurance_data)

  # Loop over the rows of the CSV file and print their values, later closing the file with the 'with' context manager.
  for row in insurance_reader:
    print(row)


# In[2]:


#Import Pandas library for data manipulation and analysis with alias 'pd'
import pandas as pd

# Load the dataset which will create a DataFrame or tabular data structure in Pandas
data = pd.read_csv('/Users/ashwinnallamothu/Desktop/Medical_Insurance/insurance.csv')

# Explore the dataset. 

#head will display first few rows of DataFrame which helps display content and structure. 
#info will provide information about the data types of the columns, the memory usage, etc. 
#describe will show summary stats like mean, std dev, min/max, etc.
print(data.head())
print(data.info())
print(data.describe())


# In[3]:


#Import necessary library to convert categorical data information into format that can be fed into machine learning algorithm. 
from sklearn.preprocessing import OneHotEncoder

# Load the data
data = pd.read_csv('/Users/ashwinnallamothu/Desktop/Medical_Insurance/insurance.csv')

# Convert categorical variables to one-hot encoding which take:
#'data': The DataFrame to be encoded
#'columns': A list of column names containing categorical variables that will be encoded.
#'drop_first' set as True will help avoid multicollinearity which can undermind statistical significance of independant variables by dropping the first category into each encoded column.
data_encoded = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)



# In[4]:


#Import matplotlib.pyplot which is a collection of functions that help matplotlib work lioke MATLAB
#Import is a data visulization library based on matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#Use pairplot to plot pairwise relationships between variables and color them with 'hue' based on values of 'smoker' column.
sns.pairplot(data, hue='smoker')
#Display the plot
plt.show()


# In[5]:


#Use Feature engineering to create new features or transform existing ones to improve the performance of a machine learning model.
#Categorize bmi data with 'pd.cut' into new column into bins to provide the model with context.
data['bmi_category'] = pd.cut(data['bmi'], bins=[0, 18.5, 24.9, 29.9, float('inf')], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])


# In[6]:


#Split data into training set and testing set.
#Predict value of variable based on value of another variable.
#Measure the average magnitude of absolute value of errors in a set of predictions.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#Split data into features and target variable
X = data_encoded.drop('charges', axis=1)
y = data_encoded['charges']

#Split data into training and testing sets.
#test_size will determine the portion of the data which will go into the test set and random state will control the random number generator used to shuffle the data before splitting.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Make predictions and evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)


# In[7]:


# Get the coefficients and feature names
coefficients = model.coef_
feature_names = X.columns

# Create a DataFrame to display the coefficients and their corresponding feature names
coeff_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort the coefficients by absolute value to see the most influential features
coeff_df['Absolute Coefficient'] = coeff_df['Coefficient'].abs()
coeff_df = coeff_df.sort_values(by='Absolute Coefficient', ascending=False)

print(coeff_df)


# In[8]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

# Train a decision tree model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Evaluate the model
dt_mae = mean_absolute_error(y_test, dt_model.predict(X_test))

plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns.tolist(), filled=True, rounded=True)
plt.show()


# In[9]:


from sklearn.ensemble import RandomForestRegressor

# Train a random forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
rf_mae = mean_absolute_error(y_test, rf_model.predict(X_test))


# In[10]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

# Train a gradient boosting model
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

# Evaluate the model
gb_mae = mean_absolute_error(y_test, gb_model.predict(X_test))

# Choose the features you want to visualize partial dependence for
features = [0, 1]  # Indices of 'age' and 'bmi' in your dataset

# Create a grid of feature values for partial dependence calculation
feature_values = np.linspace(X_train.iloc[:, features[0]].min(), X_train.iloc[:, features[0]].max(), 50)
pd_results = []

# Calculate partial dependence for the chosen features
for feature in features:
    pd_feature = []
    for value in feature_values:
        X_temp = X_train.copy()
        X_temp.iloc[:, feature] = value
        pd_feature.append(np.mean(gb_model.predict(X_temp)))
    pd_results.append(pd_feature)

# Create a plot of partial dependence
fig, ax = plt.subplots()
for i, feature in enumerate(features):
    ax.plot(feature_values, pd_results[i], label=f'Feature {feature}')
ax.set_xlabel('Feature Values')
ax.set_ylabel('Partial Dependence')
ax.legend()
plt.show()


# In[11]:


from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Assuming you have your dataset stored in X and y
# X: Features, y: Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train an SVR model
svr_model = SVR()
svr_model.fit(X_train_scaled, y_train)

# Visualize the model's predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, svr_model.predict(X_test_scaled))
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('SVR: Actual vs. Predicted')
plt.show()

# Evaluate the model
svr_mae = mean_absolute_error(y_test, svr_model.predict(X_test_scaled))
print("SVR Mean Absolute Error:", svr_mae)


# In[12]:


from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Scale the features for neural networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a neural network model
nn_model = MLPRegressor(random_state=42, max_iter=1000, early_stopping=True, validation_fraction=0.2)
nn_model.fit(X_train_scaled, y_train)

# Evaluate the model
nn_mae = mean_absolute_error(y_test, nn_model.predict(X_test_scaled))

plt.figure(figsize=(10, 6))
plt.plot(nn_model.loss_curve_)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Neural Network: Learning Curve')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, nn_model.predict(X_test_scaled))
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Neural Network: Actual vs. Predicted')
plt.show()


# In[13]:


import seaborn as sns

# Compute the correlation matrix
correlation_matrix = X.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[ ]:




