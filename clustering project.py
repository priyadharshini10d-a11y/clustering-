
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df = pd.read_csv('customers.csv')
print("Data shape:", df.shape)
print(df.head())

features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features].copy()

X = X.dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(list(K_range), inertia, marker='o')
plt.xlabel('k')
plt.ylabel('Inertia (within-cluster SSE)')
plt.title('Elbow Method')
plt.grid(True)
plt.show()

sil_scores = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores[k] = sil
print("Silhouette scores:", sil_scores)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df = df.loc[X.index]
df['Cluster'] = clusters

centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers, columns=features)
print("Cluster centers:\n", centers_df)

print(df.groupby('Cluster')[features].mean())

df.to_csv('customers_with_clusters.csv', index=False)
print("Saved customers_with_clusters.csv")

plt.figure(figsize=(7,5))
plt.scatter(df['Age'], df['Annual Income (k$)'], c=df['Cluster'], cmap='viridis', s=50)
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.title(f'Clusters (k={k}) — Age vs Annual Income')
plt.show()
