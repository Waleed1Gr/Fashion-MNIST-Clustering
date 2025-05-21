import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, mean_squared_error, confusion_matrix, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from mpl_toolkits.mplot3d import Axes3D

st.title('Fashion MNIST Unsupervised Learning Dashboard')

# Define class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def visualize_images(images, labels, class_names, n_cols=5):
    n_images = len(images)
    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    axes = axes.flatten()
    
    for i, (image, label) in enumerate(zip(images, labels)):
        axes[i].imshow(image.reshape(28, 28), cmap='gray')
        axes[i].set_title(f"{class_names[label]}")
        axes[i].axis('off')
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    st.pyplot(fig)

def search_by_class(data, labels, class_id):
    selected = data[labels == class_id]
    visualize_images(selected[:25].values, [class_id]*25, class_names)

st.sidebar.header('Upload Data')
train_file = st.sidebar.file_uploader("Upload fashion-mnist_train.csv", type="csv")
test_file = st.sidebar.file_uploader("Upload fashion-mnist_test.csv", type="csv")

if train_file and test_file:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    st.success("Files uploaded successfully!")

    # Data Preview
    st.subheader("Data Preview")
    st.write(train_df.head())
    st.write(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Split labels and features
    y_train = train_df['label']
    X_train = train_df.drop('label', axis=1)
    y_test = test_df['label']
    X_test = test_df.drop('label', axis=1)

    # Normalize
    X_train_norm = X_train / 255.0
    X_test_norm = X_test / 255.0

    # Sample images
    st.subheader("Sample Images")
    num_samples = st.slider("Number of samples", 5, 15, 10)
    fig, axes = plt.subplots(1, num_samples, figsize=(15,2))
    for i in range(num_samples):
        axes[i].imshow(X_train.iloc[i].values.reshape(28,28), cmap='gray')
        axes[i].set_title(f"Label: {class_names[y_train.iloc[i]]}")
        axes[i].axis('off')
    st.pyplot(fig)

    # Add the new class search functionality
    st.subheader("Search Images by Class")
    selected_class = st.selectbox("Select Class", range(10), format_func=lambda x: class_names[x])
    search_by_class(X_train_norm, y_train, selected_class)
# Pixel distribution
    st.subheader("Pixel Value Distribution")
    fig, ax = plt.subplots()
    sns.histplot(X_train_norm.values.flatten(), bins=30, ax=ax)
    st.pyplot(fig)

    # PCA
    st.subheader("PCA: Explained Variance")
    pca = PCA()
    pca.fit(X_train_norm.fillna(0))
    explained_var = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(explained_var)+1), explained_var)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.grid()
    st.pyplot(fig)

    # PCA Reconstruction
    st.subheader("PCA Reconstruction & MSE")
    components_list = [10, 50, 100]
    for n in components_list:
        pca_n = PCA(n_components=n)
        X_pca = pca_n.fit_transform(X_train_norm.fillna(0))
        X_recon = pca_n.inverse_transform(X_pca)
        mse = mean_squared_error(X_train_norm.fillna(0), X_recon)
        st.write(f"PCA reconstruction with {n} components: MSE = {mse:.5f}")
        fig, axes = plt.subplots(2, 3, figsize=(8,4))
        for i, idx in enumerate([0,1,2]):
            axes[0,i].imshow(X_train_norm.iloc[idx].values.reshape(28,28), cmap='gray')
            axes[0,i].set_title('Original')
            axes[0,i].axis('off')
            axes[1,i].imshow(X_recon[idx].reshape(28,28), cmap='gray')
            axes[1,i].set_title(f'Recon {n}')
            axes[1,i].axis('off')
        st.pyplot(fig)

    # SVD for clustering
    svd = TruncatedSVD(n_components=50)
    X_svd = svd.fit_transform(X_train_norm.fillna(0))

    # KMeans and DBSCAN clustering
    st.subheader("KMeans & DBSCAN Clustering (SVD 2D)")
    k = st.slider("Number of KMeans clusters", 2, 15, 10)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels_k = kmeans.fit_predict(X_svd)
    sil = silhouette_score(X_svd, labels_k)
    st.write(f"KMeans k={k}: inertia={kmeans.inertia_:.2f}, Silhouette={sil:.3f}")

    fig, ax = plt.subplots()
    scatter = ax.scatter(X_svd[:,0], X_svd[:,1], c=labels_k, cmap='tab10', s=5)
    ax.set_xlabel('SVD1')
    ax.set_ylabel('SVD2')
    ax.set_title(f'KMeans Clusters (k={k}) in SVD-2D')
    st.pyplot(fig)

    # DBSCAN
    eps = st.slider("DBSCAN eps", 1.0, 10.0, 5.0)
    dbscan = DBSCAN(eps=eps, min_samples=10)
    db_labels = dbscan.fit_predict(X_svd)
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    sil_db = silhouette_score(X_svd, db_labels) if n_clusters > 1 else -1
    st.write(f"DBSCAN (eps={eps}): Clusters={n_clusters}, Silhouette={sil_db:.3f}")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_svd[:,0], X_svd[:,1], c=db_labels, cmap='tab20', s=5)
    ax.set_xlabel('SVD1')
    ax.set_ylabel('SVD2')
    ax.set_title(f'DBSCAN Clusters (eps={eps}) in SVD-2D')
    st.pyplot(fig)

    # True labels in SVD space
    st.subheader("True Labels in SVD-2D")
    fig, ax = plt.subplots()
    ax.scatter(X_svd[:,0], X_svd[:,1], c=y_train, cmap='tab10', s=5)
    ax.set_xlabel('SVD1')
    ax.set_ylabel('SVD2')
    ax.set_title('True Labels in SVD-2D space')
    st.pyplot(fig)

    # PCA 2D
    st.subheader("KMeans & True Labels in PCA 2D")
    pca_2d = PCA(n_components=2)
    X_pca2 = pca_2d.fit_transform(X_train_norm.fillna(0))
    kmeans_2d = KMeans(n_clusters=10, random_state=42)
    labels_2d = kmeans_2d.fit_predict(X_pca2)
    fig, ax = plt.subplots()
    ax.scatter(X_pca2[:,0], X_pca2[:,1], c=labels_2d, cmap='tab10', s=8)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_title('KMeans Clusters (k=10) in PCA 2D')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.scatter(X_pca2[:,0], X_pca2[:,1], c=y_train, cmap='tab10', s=8)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_title('True Labels in PCA 2D')
    st.pyplot(fig)

    # PCA 3D
    st.subheader("KMeans & True Labels in PCA 3D")
    pca_3d = PCA(n_components=3)
    X_pca3 = pca_3d.fit_transform(X_train_norm.fillna(0))
    kmeans_3d = KMeans(n_clusters=10, random_state=42)
    labels_3d = kmeans_3d.fit_predict(X_pca3)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(X_pca3[:,0], X_pca3[:,1], X_pca3[:,2], c=labels_3d, cmap='tab10', s=8)
    ax.set_title('KMeans Clusters (k=10) in PCA 3D')
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    fig.colorbar(p)
    st.pyplot(fig)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(X_pca3[:,0], X_pca3[:,1], X_pca3[:,2], c=y_train, cmap='tab10', s=8)
    ax.set_title('True Labels in PCA 3D')
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    fig.colorbar(p)
    st.pyplot(fig)

    # Confusion matrix and purity
    st.subheader("Cluster vs. Label Analysis")
    n_clusters = 10
    kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42)
    labels_kmeans = kmeans_pca.fit_predict(X_pca2)
    cm = confusion_matrix(y_train, labels_kmeans)
    st.write("Confusion Matrix:")
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    purity_table = []
    for i in range(n_clusters):
        idx = (labels_kmeans == i)
        true_labels_in_cluster = y_train[idx]
        if len(true_labels_in_cluster) == 0:
            continue
        counts = true_labels_in_cluster.value_counts()
        most_common = counts.idxmax()
        purity = counts.max() / len(true_labels_in_cluster)
        purity_table.append({'Cluster': i, 'Majority Label': most_common, 'Purity': purity, 'Cluster Size': len(true_labels_in_cluster)})
    purity_df = pd.DataFrame(purity_table)
    st.write("Purity Table:", purity_df.sort_values('Purity', ascending=False))

    # ARI & NMI
    st.subheader("Cluster Accuracy Metrics")
    ari_kmeans = adjusted_rand_score(y_train, labels_kmeans)
    nmi_kmeans = normalized_mutual_info_score(y_train, labels_kmeans)
    st.write(f"KMeans ARI: {ari_kmeans:.3f}, NMI: {nmi_kmeans:.3f}")

    dbscan = DBSCAN(eps=5, min_samples=10)
    labels_dbscan = dbscan.fit_predict(X_pca2)
    mask = labels_dbscan != -1
    if len(set(labels_dbscan[mask])) > 1:
        ari_dbscan = adjusted_rand_score(y_train[mask], labels_dbscan[mask])
        nmi_dbscan = normalized_mutual_info_score(y_train[mask], labels_dbscan[mask])
        st.write(f"DBSCAN (ignoring noise) ARI: {ari_dbscan:.3f}, NMI: {nmi_dbscan:.3f}")
    else:
        st.write("DBSCAN found <2 clusters (after removing noise), skipping ARI/NMI.")

else:
    st.info("Please upload both train and test CSV files.")

