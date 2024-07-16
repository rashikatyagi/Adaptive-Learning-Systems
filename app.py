import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the original dataset
file_path = 'Courses.csv'
df = pd.read_csv(file_path)

# Data Preprocessing
for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].mean(), inplace=True)
df['roles'].fillna('unknown', inplace=True)
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Apply KMeans clustering with the optimal number of clusters
optimal_clusters = 7  # determined by the elbow method
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', n_init=20, max_iter=500, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[numerical_cols])
cluster_labels = kmeans.labels_

# Perform PCA for visualization
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
principal_components = pca.fit_transform(df[numerical_cols])
df_pca = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
df_pca['Cluster'] = df['Cluster']

# Plot the PCA of clustered data
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='Cluster', data=df_pca, palette='viridis')
plt.title('PCA of Clustered Data')
plt.savefig('pca_clustered_plot.png')
plt.close()

# Function to extract and summarize real-life information from the PCA plot
def extract_real_life_info(data, cluster_labels):
    unique_clusters = np.unique(cluster_labels)
    cluster_summary = {}
    
    for cluster_id in unique_clusters:
        cluster_data = data[data['Cluster'] == cluster_id]
        num_points = cluster_data.shape[0]
        
        # Extract meaningful statistics from the original data
        avg_grade = cluster_data['grade'].mean() if 'grade' in cluster_data.columns else np.nan
        avg_nplay_video = cluster_data['nplay_video'].mean()
        avg_nchapters = cluster_data['nchapters'].mean()
        avg_nforum_posts = cluster_data['nforum_posts'].mean()
        
        cluster_summary[cluster_id] = {
            'num_points': num_points,
            'avg_grade': avg_grade,
            'avg_nplay_video': avg_nplay_video,
            'avg_nchapters': avg_nchapters,
            'avg_nforum_posts': avg_nforum_posts,
        }
    
    return cluster_summary

# Extract real-life information from the clusters
cluster_summary = extract_real_life_info(df, cluster_labels)

# Streamlit UI
st.title("Adaptive Learning Systems: Student Segmentation")

st.header("1. Dataset")
st.write(df)

st.header("2. Resulting Plot After Clustering and PCA")
st.image('pca_clustered_plot.png')

st.header("3. Real-Life Information Extracted")
for cluster_id, summary in cluster_summary.items():
    st.subheader(f"Cluster {cluster_id}:")
    st.write(f"  Number of points: {summary['num_points']}")
    st.write(f"  Average grade: {summary['avg_grade']:.2f}")
    st.write(f"  Average number of videos played: {summary['avg_nplay_video']:.2f}")
    st.write(f"  Average number of chapters accessed: {summary['avg_nchapters']:.2f}")
    st.write(f"  Average number of forum posts: {summary['avg_nforum_posts']:.2f}")

st.header("Davies-Bouldin Index")
features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Cluster' in features:
    features.remove('Cluster')
X = df[features]
X_scaled = scaler.transform(X)
db_index = davies_bouldin_score(X_scaled, df['Cluster'])
st.write(f'Davies-Bouldin Index: {db_index}')
