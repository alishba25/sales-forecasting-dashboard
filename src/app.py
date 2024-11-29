import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set Streamlit page configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load the data
@st.cache_data
def load_data():
    try:
        file_path = r"C:\Users\User\OneDrive\Desktop\New folder\sales-forecasting-dashboard\data\sales_data_sample.csv"
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
        st.success("Data loaded successfully!")
        return data
    except FileNotFoundError:
        st.error(f"File not found at {file_path}. Please check the path.")
    except Exception as e:
        st.error(f"Error loading data: {e}")

# Streamlit app
st.title("Sales Forecasting Dashboard")

# Load the data
data = load_data()

if data is not None:
    st.write("Here's a preview of the data:")
    st.dataframe(data.head())



# Preprocess the data
def preprocess_data(data):
    try:
        # Convert POSTALCODE to numeric
        data['POSTALCODE'] = pd.to_numeric(data['POSTALCODE'], errors='coerce')

        # Impute missing POSTALCODE values with the median
        data['POSTALCODE'] = data['POSTALCODE'].fillna(data['POSTALCODE'].median())

        # Drop rows with missing values in critical columns
        data = data.dropna(subset=['ADDRESSLINE2', 'STATE', 'TERRITORY'], how='any')

        # Impute missing categorical columns
        data['ADDRESSLINE2'] = data['ADDRESSLINE2'].fillna('Unknown')
        data['STATE'] = data['STATE'].fillna(data['STATE'].mode()[0])
        data['TERRITORY'] = data['TERRITORY'].fillna(data['TERRITORY'].mode()[0])

        # Drop unnecessary columns
        data = data.drop(columns=['ADDRESSLINE2', 'TERRITORY'])

        # Convert ORDERDATE to datetime if exists
        if 'ORDERDATE' in data.columns:
            data['ORDERDATE'] = pd.to_datetime(data['ORDERDATE'], errors='coerce')
            # Create 'Year' and 'Month' columns
            data['Year'] = data['ORDERDATE'].dt.year
            data['Month'] = data['ORDERDATE'].dt.month

        return data
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None


# Visualizations
def create_visualizations(data):
    try:
        st.header("Data Visualizations")
        # Sales over time
        if 'ORDERDATE' in data.columns and 'SALES' in data.columns:
            st.subheader("Sales Over Time")
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=data, x='ORDERDATE', y='SALES', ci=None)
            plt.title("Sales Over Time")
            plt.xlabel("Order Date")
            plt.ylabel("Sales")
            plt.xticks(rotation=45)
            st.pyplot(plt)

        # Total Sales by Product Line
        if 'PRODUCTLINE' in data.columns:
            st.subheader("Total Sales by Product Line")
            plt.figure(figsize=(10, 6))
            sns.barplot(data=data, x='PRODUCTLINE', y='SALES', estimator=sum, ci=None)
            plt.title("Total Sales by Product Line")
            plt.xlabel("Product Line")
            plt.ylabel("Total Sales")
            plt.xticks(rotation=45)
            st.pyplot(plt)

        # Total Sales by Country
        if 'COUNTRY' in data.columns:
            st.subheader("Total Sales by Country")
            plt.figure(figsize=(12, 8))
            sns.barplot(data=data, x='COUNTRY', y='SALES', estimator=sum, ci=None)
            plt.title("Total Sales by Country")
            plt.xlabel("Country")
            plt.ylabel("Total Sales")
            plt.xticks(rotation=90)
            st.pyplot(plt)

        # Monthly Sales Heatmap
        if 'Year' in data.columns and 'Month' in data.columns:
            st.subheader("Monthly Sales Heatmap")
            monthly_sales = data.pivot_table(values='SALES', index='Year', columns='Month', aggfunc='sum')
            plt.figure(figsize=(12, 6))
            sns.heatmap(monthly_sales, annot=True, fmt=".0f", cmap="YlGnBu")
            plt.title("Monthly Sales Heatmap")
            plt.xlabel("Month")
            plt.ylabel("Year")
            st.pyplot(plt)
    except Exception as e:
        st.error(f"Error creating visualizations: {e}")


# Sales Prediction
def perform_prediction(data):
    try:
        st.header("Sales Prediction Using Linear Regression")

        # Ensure necessary columns are present
        if 'Year' in data.columns and 'Month' in data.columns and 'SALES' in data.columns:
            X = data[['Year', 'Month']]
            y = data['SALES']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predictions
            predictions = model.predict(X_test)

            # Evaluation metrics
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            st.write("Mean Squared Error:", mse)
            st.write("R-squared Score:", r2)

            # Visualization: Actual vs Predicted Sales
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, predictions, alpha=0.6)
            plt.xlabel("Actual Sales")
            plt.ylabel("Predicted Sales")
            plt.title("Actual vs Predicted Sales")
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
            st.pyplot(plt)
        else:
            st.warning("Required columns for prediction are missing!")
    except Exception as e:
        st.error(f"Error during sales prediction: {e}")


# Main app
def main():
    data = load_data()
    if data is not None:
        st.sidebar.header("Data Overview")
        st.sidebar.write(data.describe())
        st.sidebar.write(data.head())

        data = preprocess_data(data)
        if data is not None:
            create_visualizations(data)
            perform_prediction(data)


if __name__ == "__main__":
    main()
