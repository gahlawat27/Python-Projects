#!/usr/bin/env python
# coding: utf-8

# # 1. Setup and Initial Exploration

# Environment Setup

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Data Loading

# In[2]:


# Assuming the dataset is stored as 'pizza_sales.csv' in the current directory
df = pd.read_csv('C://Users//Dell//Downloads//pizza_sales.csv')


# Preliminary Data Exploration

# In[3]:


# View the first few rows of the dataset
print(df.head())

# Summary statistics for numerical columns
print(df.describe())

# Check for missing values
print(df.isnull().sum())


# # 2. Data Cleaning and Preparation# 

# Datetime Conversion

# In[4]:


# Splitting date and time columns
date_components = df['order_date'].str.split('-', expand=True)
time_components = df['order_time'].str.split(':', expand=True)

# Merging date and time components and converting to datetime
df['order_datetime'] = pd.to_datetime(date_components[0] + '-' + date_components[1] + '-' + date_components[2] + ' ' +
                                     time_components[0] + ':' + time_components[1] + ':' + time_components[2])

# Drop the original date and time columns
df.drop(['order_date', 'order_time'], axis=1, inplace=True)
print("done")


# Data Type Optimization

# In[5]:


# Convert appropriate columns to categorical type for better memory management
categorical_cols = ['pizza_id', 'order_id', 'pizza_name_id', 'pizza_size', 'pizza_category', 'pizza_name']
for col in categorical_cols:
    print(col)
    df[col] = df[col].astype('category')


# Handling Missing Data/Duplicates

# In[6]:


# Drop rows with missing values
df.dropna(inplace=True)

# Drop duplicate rows if any
df.drop_duplicates(inplace=True)
print("done")


# # 3. Exploratory Data Analysis (EDA)# 

# Sales Trends Analysis

# In[7]:


# Visualizing sales trends over time
print("in this")
plt.figure(figsize=(12, 6))
sns.lineplot(x='order_datetime', y='total_price', data=df)
plt.title('Sales Trends Over Time')
plt.xlabel('Order Date')
plt.ylabel('Total Sales')
plt.show()
print("done")


# #Performance by Category and Size

# In[8]:


# Analysis and visualization of sales performance by pizza category and size
print("in this")
plt.figure(figsize=(12, 6))
sns.barplot(x='pizza_category', y='total_price', hue='pizza_size', data=df)
plt.title('Sales Performance by Category and Size')
plt.xlabel('Pizza Category')
plt.ylabel('Total Sales')
plt.legend(title='Pizza Size')
plt.show()


# Popularity Analysis

# In[11]:


# Identification of popular and unpopular pizzas by category
popular_pizzas = df['pizza_name'].value_counts().head(10)
unpopular_pizzas = df['pizza_name'].value_counts().tail(10)

print("Top 10 Popular Pizzas:")
print(popular_pizzas)

print("\nTop 10 Unpopular Pizzas:")
print(unpopular_pizzas)

# Group by pizza name and calculate total sales
pizza_popularity = df.groupby('pizza_name')['total_price'].sum().sort_values(ascending=False)
top_10_popular = pizza_popularity.head(10)
top_10_popular.plot(kind='bar')
plt.title('Top 10 Popular Pizzas')
plt.xlabel('Pizza Name')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()


# Ingredient Analysis

# In[12]:


# Given list of ingredients
ingredients = """
Sliced Ham, Pineapple, Mozzarella Cheese
Pepperoni, Mushrooms, Red Onions, Red Peppers, Bacon
Mozzarella Cheese, Provolone Cheese, Smoked Gouda Cheese, Romano Cheese, Blue Cheese, Garlic
...
"""

# Split ingredients and count occurrences
ingredient_list = [ingredient.strip() for ingredient in ingredients.split('\n') if ingredient.strip()]
ingredient_counts = pd.Series(ingredient_list).value_counts()
ingredient_counts.plot(kind='bar')
plt.title('Ingredient Analysis')
plt.xlabel('Ingredient')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()


# # 4. Recommendation System Development:# 

# In[14]:


# Data Preparation
# Assume we have a DataFrame 'df' with columns 'pizza_name' and 'pizza_ingredients'
# Encode pizza ingredients
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
encoded_ingredients = pd.DataFrame(mlb.fit_transform(df['pizza_ingredients'].str.split(', ')), columns=mlb.classes_, index=df.index)

# Concatenate encoded ingredients with original DataFrame
df_encoded = pd.concat([df, encoded_ingredients], axis=1)

# Recommendation Algorithm (Simple Example: Based on Ingredient Similarity)
def recommend_pizza(ingredients):
    ingredient_vector = mlb.transform([ingredients])
    ingredient_similarity = df_encoded[df_encoded['pizza_name'] != ingredients][encoded_ingredients.columns].mul(ingredient_vector).sum(axis=1)
    recommended_pizzas = df_encoded.loc[ingredient_similarity.nlargest(5).index]['pizza_name'].unique()
    return recommended_pizzas

# Example Usage:
print(recommend_pizza('Pepperoni, Mushrooms, Red Onions, Red Peppers, Bacon'))

# System Evaluation and Insights & Actionable Recommendations
# These sections would remain the same, focusing on evaluating the recommendation system's effectiveness and deriving insights and actionable recommendations based on the analysis.


# # Insights and Actionable Recommendations

# Key Insights:
# Popularity Analysis: The top 10 popular pizzas include classics like "The Classic Deluxe Pizza" and "The Pepperoni Pizza," indicating a preference for traditional flavors. However, more unique options like "The Barbecue Chicken Pizza" and "The Thai Chicken Pizza" also demonstrate popularity, suggesting a demand for diverse flavor profiles.
# 
# Recommendation System: The recommendation system suggests pizzas that are similar to the popular choices, such as "The Barbecue Chicken Pizza," "The Thai Chicken Pizza," and "The Southwest Chicken Pizza." This indicates a preference for chicken-based pizzas with distinctive flavors like barbecue and Thai.
# 
# Sales Trends: Further analysis of sales trends over different time frames could provide insights into seasonal variations in pizza preferences. For example, certain pizzas might be more popular during holidays or special events.
# 
# Actionable Recommendations:
# Menu Optimization: Consider leveraging the popularity of classic pizzas while introducing new, innovative flavors to cater to diverse tastes. Experiment with limited-time offers or seasonal specials to gauge customer interest in new options.
# 
# Promotional Strategies: Develop targeted marketing campaigns to promote popular pizzas and drive sales. Highlight unique flavor combinations or limited-time offers to create excitement and attract customers.
# 
# Customer Engagement: Encourage customer feedback and engagement to understand preferences better. Conduct surveys or offer incentives for feedback to gather insights into customer preferences and tailor the menu accordingly.

# # Conclusion and Future Work

# Conclusion:
# The analysis provides valuable insights into pizza sales trends, popular choices, and recommendations for menu optimization. By understanding customer preferences and leveraging data-driven insights, the pizzeria can enhance its menu offerings, drive sales, and improve customer satisfaction.
# 
# Future Directions:
# Customer Segmentation: Explore customer segmentation based on demographics, preferences, and purchase behavior to personalize offerings and marketing strategies.
# 
# Menu Experimentation: Continuously experiment with new flavors, ingredients, and menu combinations to stay innovative and meet evolving customer preferences.
# 
# Integration of Technology: Explore opportunities to leverage technology, such as online ordering platforms or mobile apps, to streamline the ordering process, enhance customer experience, and gather more comprehensive data for analysis.
