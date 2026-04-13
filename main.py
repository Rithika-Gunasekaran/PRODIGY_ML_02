import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('Mall_Customers.csv')

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.show()

# Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Plot clusters
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y_kmeans)
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.show()

# Apply KMeans with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Show clusters in graph
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y_kmeans)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segments')
plt.show()

# Save output file
df['Cluster'] = y_kmeans
df.to_csv('output.csv', index=False)
