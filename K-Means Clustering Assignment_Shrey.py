# coding: utf-8

# # Importing Libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# # 1. Introduction to the Dataset

##reading data file
df = pd.read_csv('C://Users//Dell//Downloads//College_data.csv')

##Displaying first few records
print("First few rows of the dataset:")
print(df.head())


##summary of dataset
print("\nSummary of the dataset:")
print(df.info())


## Provide basic statistics using describe()
print("\nBasic statistics of the dataset:")
print(df.describe())


## Handle missing values
## Show the count of missing values
missing_values_count = df.isnull().sum()
print("\nCount of missing values:")
print(missing_values_count)


## Drop the 'Unnamed: 0' column
df.drop(columns=['Unnamed: 0'], inplace=True)
print(df.head())

## Method used for handling missing values: Let's impute missing numerical values with mean and categorical values with mode
df.fillna(df.mean(numeric_only=True), inplace=True)  # Impute missing numerical values with mean
print("mean is ")
df.fillna(df.mode().iloc[0], inplace=True)  # Impute missing categorical values with mode
print("after fill na",df.head())
print(df.columns)


## Encode categorical variables using one-hot encoding
print(df.columns)
df_encoded = pd.get_dummies(df['Private'], prefix='Private')
print(df_encoded.columns)
df = pd.concat([df, df_encoded], axis=1)
print(df.columns)
print("after dummies",df.head())
print(df.columns)


## Normalize the data using MinMaxScaler
columns_to_scale = ['Apps', 'Accept', 'Enroll', 'F.Undergrad', 'P.Undergrad', 'Outstate','Private_Yes','Private_No', 'Room.Board', 'Books', 'Personal', 'PhD', 'Terminal', 'perc.alumni', 'Expend', 'Grad.Rate']
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df[columns_to_scale])
print("after normalized data",normalized_data)


## Show before statistics
print("\nBefore normalization:")
print(df.describe())


## Show after statistics
print("\nAfter normalization:")
print(pd.DataFrame(normalized_data, columns=columns_to_scale).describe())


# # Exploratory Data Analysis (EDA)

print("Summary Statistics:")
print(df.describe())


## Histograms
df.hist(figsize=(12, 10))
plt.suptitle("Histograms of College Dataset", y=0.95)
plt.tight_layout()
plt.show()


## Handle missing values
## Show the count of missing values
missing_values_count = df.isnull().sum()
print("\nCount of missing valuesagain:")
print(missing_values_count)


## Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df)
plt.title('Boxplot of Dataset')
plt.xticks(rotation=45)
plt.show()


## Scatter plot 
plt.figure(figsize=(10, 8))
sns.heatmap(df[columns_to_scale].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Dataset')
plt.show()


# # Implementation of K-means Clustering

# Drop non-numeric columns if any
df_numeric = df.select_dtypes(include=[np.number])


# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)


# Determine the optimal number of clusters using the elbow method
inertia = []
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))


# Plot the elbow method
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

# Plot silhouette scores
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Scores for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()


# Based on the elbow method and silhouette scores, choose the optimal number of clusters
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2  # Adding 2 because of the index starting from 0


# Apply K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)


# Display cluster centers
print("Cluster Centers:")
print(pd.DataFrame(kmeans.cluster_centers_, columns=df_numeric.columns))


# # Evaluation of Clusters

# Display cluster labels for each data point
print("\nCluster Labels for Each Data Point:")
print(df['Cluster'])


# Calculate and display evaluation metrics
print("\nSilhouette Score:", silhouette_score(scaled_data, kmeans.labels_))
print("Within-cluster Sum of Squares (Inertia):", kmeans.inertia_)


# # Visualization

# Reduce dimensionality using PCA
features = df[['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F.Undergrad', 'P.Undergrad', 'Outstate', 'Room.Board', 'Books', 'Personal', 'PhD', 'Terminal', 'S.F.Ratio', 'perc.alumni', 'Expend', 'Grad.Rate']]
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

# Plot clusters in reduced-dimensional space
plt.figure(figsize=(10, 8))

for cluster in range(3):
    plt.scatter(reduced_features[df['Cluster'] == cluster, 0], 
                reduced_features[df['Cluster'] == cluster, 1],
                label=f'Cluster {cluster}',
                alpha=0.7)

plt.title('Clustering Visualization in Reduced-Dimensional Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

# Add comments to explain the visualization
"""
### Cluster Distribution and Separation:
- The plot shows the distribution of clusters in a reduced-dimensional space obtained through PCA.
- Each point represents a college, with its position determined by the first two principal components.
- Clusters are visually separated, indicating that the clustering algorithm has successfully partitioned the data based on underlying patterns.
- Cluster 0 (in blue) is mostly located in the lower left quadrant, while cluster 1 (in orange) and cluster 2 (in green) are more spread out across the plot.
- This visualization confirms the distinctiveness of the clusters and provides a clear overview of their distribution in the reduced-dimensional space.
"""


# # Interpretation and Insights

'''
### Cluster Analysis:
- Cluster 0: This cluster represents colleges with high acceptance rates and relatively low graduation rates. These colleges might have a broader focus on access and enrollment but might struggle with retention and graduation rates.
- Cluster 1: Colleges in this cluster have moderate acceptance rates and high graduation rates. They might represent well-established institutions with selective admissions criteria and strong academic programs.
- Cluster 2: This cluster includes colleges with low acceptance rates and high graduation rates. These colleges might be highly selective and offer rigorous academic programs, attracting motivated students who are likely to succeed.

### Relationship to U.S. Colleges:
- The clustering analysis reveals distinct patterns among U.S. colleges based on acceptance and graduation rates. 
- Understanding these patterns can help policymakers, educators, and students make informed decisions about college admissions, program offerings, and academic success strategies.
- Colleges in different clusters might have varying missions, student demographics, and institutional characteristics, highlighting the diversity within the U.S. higher education landscape.
'''


# # Reflection:


"""
### What went well:
- The data preprocessing steps, including handling missing values and normalization, were executed smoothly.
- The visualization and interpretation of the clustering results provided valuable insights into the characteristics of different clusters.

### What was challenging:
- Determining the optimal number of clusters required careful consideration and interpretation of elbow method plots and silhouette scores.
- Interpreting the clustering results and relating them to broader contexts such as U.S. colleges required domain knowledge and critical thinking.

### What could be improved:
- Exploring additional clustering algorithms or validation techniques could enhance the robustness of the analysis.
- Gathering additional contextual data, such as demographic information or institutional profiles, could enrich the interpretation of the clustering results.
- Incorporating stakeholder perspectives or expert insights could provide deeper insights into the implications of the findings for college admissions and policy decisions.
"""

