#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Loading the dataset
data = pd.read_csv('sales_data_sample.csv', encoding='ISO-8859-1')


# In[3]:


# Displaying first few rows to understand the data
data.head()


# In[4]:


# Drop rows with missing values in critical columns
data = data.dropna(subset=['ADDRESSLINE2', 'STATE', 'TERRITORY'], how='any')

# Impute missing values for numerical columns (e.g., POSTALCODE)
data['POSTALCODE'] = data['POSTALCODE'].fillna(data['POSTALCODE'].median())

# Impute missing categorical columns (e.g., ADDRESSLINE2, STATE)
data['ADDRESSLINE2'] = data['ADDRESSLINE2'].fillna('Unknown')
data['STATE'] = data['STATE'].fillna(data['STATE'].mode()[0])
data['TERRITORY'] = data['TERRITORY'].fillna(data['TERRITORY'].mode()[0])

# Drop columns with too many missing values if not critical
data = data.drop(columns=['ADDRESSLINE2', 'TERRITORY'])

# Final check for missing values
print(f"Missing values after cleaning:\n{data.isnull().sum()}")


# In[5]:


# Drop rows with missing values
data.dropna(inplace=True)

# Convert date columns to DateTime format (if applicable)
if 'ORDERDATE' in data.columns:
    data['ORDERDATE'] = pd.to_datetime(data['ORDERDATE'])
    
# Creating 'Year' and 'Month' columns for better time-based analysis
if 'ORDERDATE' in data.columns:
    data['Year'] = data['ORDERDATE'].dt.year
    data['Month'] = data['ORDERDATE'].dt.month


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns

# Sales over time
plt.figure(figsize=(12,6))
sns.lineplot(data=data, x='ORDERDATE', y='SALES')
plt.title("Sales Over Time")
plt.show()


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot sales over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='ORDERDATE', y='SALES', ci=None)
plt.title("Sales Over Time")
plt.xlabel("Order Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.show()


# In[8]:


# Check if 'PRODUCTLINE' column exists
if 'PRODUCTLINE' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x='PRODUCTLINE', y='SALES', estimator=sum, ci=None)
    plt.title("Total Sales by Product Line")
    plt.xlabel("Product Line")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.show()


# In[9]:


# Check if 'COUNTRY' column exists
if 'COUNTRY' in data.columns:
    plt.figure(figsize=(12, 8))
    sns.barplot(data=data, x='COUNTRY', y='SALES', estimator=sum, ci=None)
    plt.title("Total Sales by Country")
    plt.xlabel("Country")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=90)
    plt.show()


# In[10]:


# Create pivot table for heatmap
if 'Year' in data.columns and 'Month' in data.columns:
    monthly_sales = data.pivot_table(values='SALES', index='Year', columns='Month', aggfunc='sum')
    plt.figure(figsize=(12, 6))
    sns.heatmap(monthly_sales, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title("Monthly Sales Heatmap")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.show()


# In[14]:


# Step 1: Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 2: Prepare data (replace 'Year' and 'Month' with columns from your dataset as needed)
X = data[['Year', 'Month']]  # Ensure 'Year' and 'Month' columns are in the dataset
y = data['SALES']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
predictions = model.predict(X_test)

# Step 6: Calculate Mean Squared Error and R-squared Score
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Step 7: Visualize Actual vs Predicted Sales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.6)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Line for perfect predictions
plt.show()


# In[ ]:




