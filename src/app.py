#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import plotly.express as px
data = pd.read_csv('sales_data_sample.csv', encoding='ISO-8859-1')
st.title("Sales Data Analysis Dashboard")

# Plot sales over time
fig = px.line(data, x='OrderDate', y='SALES', title='Sales Over Time')
st.plotly_chart(fig)

# Plot sales by product line
if 'PRODUCTLINE' in data.columns:
    product_sales = data.groupby('PRODUCTLINE')['SALES'].sum().reset_index()
    fig2 = px.bar(product_sales, x='PRODUCTLINE', y='SALES', title='Sales by Product Line')
    st.plotly_chart(fig2)

# Plot sales by country
if 'COUNTRY' in data.columns:
    country_sales = data.groupby('COUNTRY')['SALES'].sum().reseta_index()
    fig3 = px.bar(country_sales, x='COUNTRY', y='SALES', title='Sales by Country')
    st.plotly_chart(fig3)

# Forecasting
st.subheader("Sales Forecasting")
forecast_year = st.slider("Select year for forecasting", min_value=2023, max_value=2030)
forecast_month = st.slider("Select month for forecasting", min_value=1, max_value=12)
forecast_sales = model.predict([[forecast_year, forecast_month]])
st.write(f"Predicted Sales for {forecast_month}/{forecast_year}: ${forecast_sales[0]:,.2f}")
s


# In[ ]:




